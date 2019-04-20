import h5py
import numpy as np
import os
from math import *
import sys
import torch
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self,X,Y,grid):
        self.X = X
        self.Y = Y
        self.grid = grid

    def __getitem__(self,idx):
        return (self.X[idx],self.Y[idx],self.grid[idx])

    def __len__(self):
        return len(self.X)

class CFDdata:
    def __init__(self, fileName, train_size=6000,target='P'):
        self.getInputOutput(fileName,train_size,True,target)

    def get(self,kind):
        if (kind == 'train'):
            return Data(self.train_inputData,self.train_outputData,self.train_grid)
        elif (kind == 'validation'):
            return Data(self.val_inputData,self.val_outputData,self.val_grid)
        elif (kind == 'test'):
            return Data(self.test_inputData,self.test_outputData,self.test_grid)
        else:
            print('Error! Unknown data kind (train, validation, test)')
            exit()

    def getInputOutput(self,fileName,train_size=6000,shuffle=True,target='P'):
        print('Target=',target)
        h5 = h5py.File(fileName, 'r')
        nData = len(h5)
        shape = h5['dataset00001']['U'].shape
        n = 1
        if (target=='all'):
            n = 4
        inputData = np.zeros((nData,6,shape[0]-1,shape[1]-1))
        outputData = np.zeros((nData,n,shape[0]-1,shape[1]-1))
        gridPoints = np.zeros((nData,2,shape[0]-1,shape[1]-1))
        index = np.arange(0,nData,dtype=int)
        print("Data size = {}".format(nData))
        if shuffle:
            np.random.shuffle(index)

        for i in index:
            g = h5['dataset%05d' % (i+1)]
            inputData[i,0,:,:] = np.array(g['Uinput'])[:128,:32]
            inputData[i,1,:,:] = np.array(g['Vinput'])[:128,:32]
            inputData[i,2,:,:] = np.array(g['Mach'])[:128,:32]
            inputData[i,3,:,:] = np.array(g['phi'])[:128,:32]
            inputData[i,4,:,:] = np.array(g['gradPhi-X'])[:128,:32]
            inputData[i,5,:,:] = np.array(g['gradPhi-Y'])[:128,:32]
            if (target == 'all'):
                outputData[i,0,:,:] = np.array(g['P'])[:128,:32]
                outputData[i,1,:,:] = np.array(g['Ro'])[:128,:32]
                outputData[i,2,:,:] = np.array(g['U'])[:128,:32]
                outputData[i,3,:,:] = np.array(g['V'])[:128,:32]
            else:
                outputData[i,0,:,:] = np.array(g[target])[:128,:32]
            gridPoints[i,0,:,:] = np.array(g['X'])[:128,:32]
            gridPoints[i,1,:,:] = np.array(g['Y'])[:128,:32]

        validate_size = (nData - train_size) // 2

        self.train_inputData = inputData[0:train_size]
        self.train_outputData = outputData[0:train_size]
        self.train_grid = gridPoints[0:train_size]

        self.val_inputData = inputData[train_size:train_size+validate_size]
        self.val_outputData = outputData[train_size:train_size+validate_size]
        self.val_grid = gridPoints[train_size:train_size+validate_size]

        self.test_inputData = inputData[train_size+validate_size:]
        self.test_outputData = outputData[train_size+validate_size:]
        self.test_grid = gridPoints[train_size+validate_size:]

        # return train_inputData,train_outputData,val_inputData,val_outputData,test_inputData,test_outputData
