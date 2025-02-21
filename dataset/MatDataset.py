import os
# os.environ['HDF5_DISABLE_VERSION_CHECK'] = '2' # Only add this for TRACE to work, comment out for other cases! 

import torch
import numpy as np
import scipy.io
import ctypes
import h5py
import shutil
# import pyJHTDB
# from pyJHTDB import libJHTDB
import sklearn.metrics
# from torch_geometric.data import Data, InMemoryDataset
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
# from paraview.simple import *
# from dolfin import *


class Sub_JHTDB(Dataset):
    '''
    Includes a subset of the JHTDB dataset given the indices of the data
    '''
    def __init__(self, root, indices):
        self.root = root
        # verify that JHTDB data is correctly processed
        if not os.path.exists(os.path.join(self.root, 'processed', 'data.pt')):
            raise ValueError('JHTDB data is not processed yet')
        self.indices = indices

        self.data = torch.load(os.path.join(self.root, 'processed', 'data.pt'))
        self.data = [self.data[i] for i in self.indices]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    