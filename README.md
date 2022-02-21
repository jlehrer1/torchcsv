# torchcsv 
An Pytorch Dataset and Dataloader library for handling numerical data too large to fit in local memory.

## Usage
The `CSVDataset` class inherits from `torch.Dataset` like we always do with custom Dataset classes. However, rather than reading the entire data and label `.csv` into memory, we make two assumptions:
1. The dataset is too large to fit in local memory 
2. The labels are contained in a separate file. If this isn't the case, consider using `Dask` to obtain the column of interest, and then continue.

So, we initialize the `CSVDataset` object as 
```python 
from torchcsv import CSVDataset 

data = CSVDataset(
    datafile='path/to/datafile.csv',
    labelfile='path/to/labelfile.csv',
    target_label='Animal Type', # Column name containing targets in labelfile.csv
    # indices=idx_list # Optionally, pass a list of purely numeric indices to use instead of the entire indices of the labelfile 
)
```
For example, getting a 16.3 dimensional sample takes 
```python
> %%time
> test[1]
CPU times: user 5.99 ms, sys: 576 µs, total: 6.56 ms
Wall time: 6.19 ms
(tensor([0., 0., 0.,  ..., 0., 0., 0.]), 16)
```
Now, we can use this like a regular PyTorch Dataset, but without having to worry about memory issues!

For example,
```python
from torch.utils.data import Dataloader 
data = DataLoader(data, batch_size=4, num_workers=0)
```
Gives us that 

```python 
%%time 
next(iter(test))

CPU times: user 25.6 ms, sys: 20.8 ms, total: 46.4 ms
Wall time: 76.9 ms

[tensor([[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 1.2663,  ..., 0.0000, 0.0000, 0.0000]]),
 tensor([16, 16,  4,  4])]
```

So loading a minibatch of size 4 takes about a quarter of a second. The `CSVDataset` class should be scalable, and will keep in memory what it can via the `linecache` library. 