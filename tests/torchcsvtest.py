from torchcsv import CSVDataset
import pandas as pd 
import numpy as np 
import pathlib, os 

here = pathlib.Path(__file__).parent.resolve()

# Generate synthetic binary data
print('Generating synethic data')
df = pd.DataFrame(np.random.rand(100, 100))
labels = pd.DataFrame(np.random.randint(0, 2, size=(100, 1)))

map = lambda x: 'Cat' if x else 'Dog'
labels.loc[:, 0] = labels.loc[:, 0].apply(map)

df.to_csv(os.path.join(here, 'test.csv'), index=False)
labels.to_csv(os.path.join(here, 'labels.csv'), index=False)


print(pd.read_csv(os.path.join(here, 'test.csv')).head())
print(pd.read_csv(os.path.join(here, 'labels.csv')).head())

dataset = CSVDataset(
    datafile=os.path.join(here, 'test.csv'),
    labelfile=os.path.join(here, 'labels.csv'),
    target_label='0',
    quiet=True 
)

for i in range(0, 10):
    print(dataset[i])
    print(labels.loc[:, 0].iloc[i])
    print(' ')