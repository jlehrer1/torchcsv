import torch 
import numpy as np 
import csv 
import pandas as pd 
import linecache

from typing import *
from functools import cached_property
from sklearn.utils.class_weight import compute_class_weight

class CSVDataset(torch.utils.data.Dataset):
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
        filename: str, 
        labelname: str, 
        class_label: str,
        indices: Iterable[int]=None,
        skip=3,
        cast=True,
        index_col='cell',
        normalize=False,
    ):
        self.filename = filename
        self.labelname = labelname # alias 
        self._class_label = class_label
        self._index_col = index_col 
        self.skip = skip
        self.cast = cast
        self.normalize = normalize 
        self.indices = indices

        if indices is None:
            self._labeldf = pd.read_csv(labelname).reset_index(drop=True)
        else:
            self._labeldf = pd.read_csv(labelname).loc[indices, :].reset_index(drop=True)

    def __getitem__(self, idx):
        # Handle slicing 
        if isinstance(idx, slice):
            if idx.start is None or idx.stop is None:
                raise ValueError(f"Error: Unlike other iterables, {self.__class__.__name__} does not support unbounded slicing since samples are being read as needed from disk, which may result in memory errors.")

            step = (1 if idx.step is None else idx.step)
            idxs = range(idx.start, idx.stop, step)
            return [self[i] for i in idxs]

        # The label dataframe contains both its natural integer index, as well as a "cell" index which contains the indices of the data that we 
        # haven't dropped. This is because some labels we don't want to use, i.e. the ones with "Exclude" or "Low Quality".
        # Since we are grabbing lines from a raw file, we have to keep the original indices of interest, even though the length
        # of the label dataframe is smaller than the original index

        # The actual line in the datafile to get, corresponding to the number in the self._index_col values 
        data_index = self._labeldf.loc[idx, self._index_col]

        # get gene expression for current cell from csv file
        # We skip some lines because we're reading directly from 
        line = linecache.getline(self.filename, data_index + self.skip)
        
        if self.cast:
            data = torch.from_numpy(np.array(line.split(','), dtype=np.float32)).float()
        else:
            data = np.array(line.split(','))

        if self.normalize:
            data = data / data.max()

        label = self._labeldf.loc[idx, self._class_label]

        return data, label

    def __len__(self):
        return len(self._labeldf) # number of total samples 

    @property
    def columns(self): # Just an alias...
        return self.features

    @cached_property # Worth caching, since this is a list comprehension on up to 50k strings. Annoying. 
    def features(self):
        data = linecache.getline(self.filename, self.skip - 1)
        data = [x.split('|')[0].upper().strip() for x in data.split(',')]

        return data

    @cached_property
    def labels(self):
        return self._labeldf.loc[:, self._class_label].unique()

    @property
    def shape(self):
        return (self.__len__(), len(self.features))

    @cached_property 
    def class_weights(self):
        labels = self._labeldf.loc[:, self._class_label].values

        return compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(filename={self.filename}, labelname={self.labelname})"

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(filename={self.filename}, "
            f"labelname={self.labelname}, "
            f"skip={self.skip}, "
            f"cast={self.cast}, "
            f"normalize={self.normalize}, "
            f"indices={self.indices})"
        )