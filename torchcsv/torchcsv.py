import torch 
import numpy as np 
import csv 
import pandas as pd 
import linecache

from typing import *

class CSVDataset(torch.Dataset):
    """
    Defines a PyTorch Dataset for a CSV too large to fit in memory. 

    Init params:
    datafile: Path to csv data file, where rows are samples and columns are features
    labelfile: Path to label file, where column '# labels' defines classification labels
    target_label: Label to train on, must be in labelfile file 
    indices=None: List of indices to use in dataset. If None, all indices given in labelfile are used.
    """

    def __init__(
        self, 
        datafile: str, 
        labelfile: str, 
        target_label: str,
        indices: Iterable[int]=None,
        skip=2,
        quiet=False
    ):
        self._datafile = datafile
        if not quiet: print('Reading label file into memory')
        self._labelfile = (pd.read_csv(labelfile) if indices is None else pd.read_csv(labelfile).iloc[indices, :])
        self._total_data = 0
        self._target_label = target_label

        self.index = indices
        self.skip=skip
        self.columns = self.get_features()

    def __getitem__(self, idx):
        # Get index in dataframe from integer index
        idx = self._labelfile.iloc[idx].name
        
        # Get label
        label = self._labelfile.loc[idx, self._target_label]
        
        # get gene expression for current cell from csv file
        # index + 2: Header is blank line and zeroth row is column names
        line = linecache.getline(self._datafile, idx + self.skip)
        csv_data = csv.reader([line])
        data = [x for x in csv_data][0]
        
        return torch.from_numpy(np.array([float(x) for x in data])).float(), label
    
    def __len__(self):
        return self._labelfile.shape[0] # number of total samples 
    
    def num_labels(self):
        return self._labelfile[self._target_label].nunique()
    
    def num_features(self):
        return len(self.__getitem__(0)[0])

    def get_features(self):
        line = linecache.getline(self._datafile, 0)
        csv_data = csv.reader([line])
        data = [x for x in csv_data][0]

        return data
