# %%
import torch
from torch_geometric.nn import MLP, fps, global_max_pool, radius
from torch_geometric.nn.conv import PointNetConv

# %% Implementing the PointNet++ Model using PyTorch Geometric
class SetAbstraction(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]

        return x, pos, batch

# %%
class GlobalSetAbstraction(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)

        return x, pos, batch

# %%
class PointNet2(torch.nn.Module):
    def __init__(
        self,
        set_abstraction_ratio_1, set_abstraction_ratio_2,
        set_abstraction_radius_1, set_abstraction_radius_2, dropout, n_classes=10
    ):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SetAbstraction(
            set_abstraction_ratio_1,
            set_abstraction_radius_1,
            MLP([3, 64, 64, 128])
        )
        self.sa2_module = SetAbstraction(
            set_abstraction_ratio_2,
            set_abstraction_radius_2,
            MLP([128 + 3, 128, 128, 256])
        )
        self.sa3_module = GlobalSetAbstraction(MLP([256 + 3, 256, 512, 1024]))

        self.mlp = MLP([1024, 512, 256, n_classes], dropout=dropout, norm=None)

    def forward(self, data):
        sa0_out = (data.x, data.pos, data.batch) # Note how the batch is handled
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        return self.mlp(x).log_softmax(dim=-1)

# %% Test out the model by running it on a single batch of point cloud data
if __name__=='__main__':
    import torch_geometric.transforms as T
    from torch_geometric.datasets import ModelNet
    from torch_geometric.loader import DataLoader
    from paths import DATA

    dataset = ModelNet(root=DATA,
                       train=True,
                       transform=None, # T.SamplePoints(1024),
                       pre_transform=T.NormalizeScale()).shuffle()[:100]

    loader = DataLoader(dataset,
                        batch_size=16,
                        shuffle=True,
                        num_workers=4)

    for batch in loader:
        print('batch info  : ', batch)
        break

    model = PointNet2(0.748, 0.4817, 0.3316, 0.2447, 0.1, n_classes=10) # Retrieved after parameter sweep

    out = model(batch)

    print('batch info  : ', batch)
    print('output shape: ', out.shape)
