from paths import DATA
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet

ModelNet(root=DATA,  train=True, name='10', pre_transform=T.NormalizeScale())
ModelNet(root=DATA,  train=False, name='10', pre_transform=T.NormalizeScale())