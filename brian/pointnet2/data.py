# %%
from paths import DATA

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader

# %%
data = ModelNet(root=DATA, name='10')

dataset_train = ModelNet(root               = DATA, 
                         name               = '10', 
                         train              = True,
                         transform          = T.SamplePoints(1024),
                         pre_transform      = T.NormalizeScale())

dataloader_train = DataLoader(dataset        = dataset_train,
                              batch_size     = 32,
                              shuffle        = True,
                              num_workers    = 1,
                              pin_memory     = True) # pin_memory=True to keep the data in GPU

for batch in dataloader_train:
    print(batch)
    break

print(batch)
print('xyz coordinates of the points [batch.pos]', batch.pos)
print('labels [batch.y]', batch.y)
print('batch index [batch.batch]', batch.batch)

# %% Sample data visualizations

# %% See how many data points are in each class
