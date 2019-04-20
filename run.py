import sys
# sys.path.append('./')
from Unet import *

run(batch_size=60,
    training_size=6500,
    num_epochs=1000,
    hdf5file="CFDdata_final.hdf5",
    depth=4,
    nFilters=256,
    target='all',
    writeEvery=10,
    lr=0.001)
