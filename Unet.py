import torch
from readCFDdata import *
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os
import time

def run(batch_size=50,training_size=6000,num_epochs=100,hdf5file="CFDdata_final.hdf5",depth=4,nFilters=256,target='all',lr=0.0001,writeEvery=10):

    data = CFDdata(hdf5file,train_size=training_size,target=target)

    train_data = data.get(kind='train')
    val_data = data.get(kind='validation')
    test_data = data.get(kind='test')

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                               batch_size=batch_size,
                                               shuffle=True);

    val_loader = torch.utils.data.DataLoader(dataset=val_data,
                                              batch_size=batch_size,
                                              shuffle=True);

    test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                              batch_size=batch_size,
                                              shuffle=True);

    n = 1
    if (target == 'all'):
        n = 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UNet(in_channels=6, out_channels=n, depth=depth, nf_start=nFilters)
    model = nn.DataParallel(model).to(device)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of param= ',pytorch_total_params)

    params = list(model.parameters())
    #loss function and optimizer
    criterion = nn.L1Loss();
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.95, nesterov=True)

    losses = []
    lossesPerEpoch = []
    accuracyPerEpoch = []
    print("Training size = {}".format(len(train_data)))

    fid = open('convergence.dat','w')
    rms = 1e6

    start = time.time()

    bestValidationLoss = 1e6

    for epoch in range(num_epochs):
        lossesPerEpoch.append(trainModel(model,train_loader,criterion,batch_size,device,optimizer,writeEvery))
        valLoss = evalLoss(model,train_loader,nn.L1Loss(),device)
        fid.write("%04d %12.12le %12.12le %12.12le %12.12le\n" % (epoch+1, time.time() - start, lossesPerEpoch[-1],valLoss,rms))
        fid.flush()
        print("Epoch %d/%d PhysTime: %12.12le Loss: %12.12le ValLoss: %12.12le" % (epoch+1,num_epochs,time.time()-start,lossesPerEpoch[-1],valLoss))

        if (valLoss < bestValidationLoss):
            bestValidationLoss = valLoss
            saveModel(model,"best_model.pth")
            rms = sqrt(evalLoss(model,test_loader,nn.MSELoss(),device))
            print("RMS = {:12.12e}".format(rms))

    fid.close()

    loadModel(model,"best_model.pth")
    rms = sqrt(evalLoss(model,test_loader,nn.MSELoss(),device))
    print("RMS = {:12.12e}".format(rms))

def trainModel(model, train_loader, criterion, batch_size, device, optimizer, writeEvery):
    model.train()
    Loss = 0
    for i, (images, labels, grids) in enumerate(train_loader):
        images = Variable(images.float())
        labels = Variable(labels.float()).to(device)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        Loss = loss.item();

        if (i+1) % writeEvery == 0:
            writeFlowField(outputs,labels,grids,i)
            print ('\t\tIter : %d/%d,  Loss: %.4f'
                   %(i+1, len(train_loader), Loss))
    return Loss

def evalLoss(model, data_loader, criterion, device):
    model.eval()

    count = 0
    loss = 0.0

    with torch.no_grad():
        for id, (images, labels, grids) in enumerate(data_loader):
            images = Variable(images.float())
            labels = Variable(labels.float()).to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            count += 1

    loss = (loss / count)
    return loss

def saveModel(model,PATH):
    torch.save(model.state_dict(),PATH)

def loadModel(model,PATH):
    model.load_state_dict(torch.load(PATH))

def writeFlowField(prediction,expected,grid,id):
    try:
        os.mkdir("TrainingSolution")
    except:
        pass
    fid = open("TrainingSolution/test%05d.dat" % (id),'w')
    fid.write("VARIABLES= \"X\",\"Y\",\"predicted\",\"expected\"\n")
    ni = prediction.size(2)
    nj = prediction.size(3)
    fid.write("ZONE T=\"SOL\", I=%d, J=%d\n" % (ni+1,nj))
    for j in range(nj):
        for i in range(ni):
            fid.write("%12.24lf %12.24lf %12.24lf %12.24lf\n" % (grid[0,0,i,j],grid[0,1,i,j],prediction[0,0,i,j],expected[0,0,i,j]))
        fid.write("%12.24lf %12.24lf %12.24lf %12.24lf\n" % (grid[0,0,0,j],grid[0,1,0,j],prediction[0,0,0,j],expected[0,0,0,j]))
    fid.close()

class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=4, depth=4, nf_start=256):

        super(UNet, self).__init__()

        self.depth = depth
        inChannels = in_channels
        ouChannels = in_channels
        self.downSizing = nn.ModuleList()
        self.upSizing = nn.ModuleList()

        for i in range(depth):
            self.downSizing.append(ConvBlock(inChannels,ouChannels))
            inChannels = ouChannels
            ouChannels *= 2

        ouChannels = inChannels // 2
        for i in reversed(range(depth - 1)):
            self.upSizing.append(UpBlock(inChannels, ouChannels))
            inChannels = ouChannels
            ouChannels = inChannels // 2

        self.out = nn.Conv2d(inChannels, out_channels, kernel_size=1)

    def forward(self, x):
        x_saved = []
        for i, down in enumerate(self.downSizing):
            x = down(x)
            if i < len(self.downSizing)-1:
                x_saved.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.upSizing):
            x = up(x, x_saved[-i-1])

        return self.out(x)

class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
        nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_size),
        nn.Conv2d(out_size, out_size, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_size)
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UpBlock, self).__init__()

        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_size, out_size)

    def forward(self, x, x_prev):
        up = self.up(x)
        out = torch.cat([up, x_prev], 1)
        return self.conv(out)
