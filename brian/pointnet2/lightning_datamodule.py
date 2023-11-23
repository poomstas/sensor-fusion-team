# %%
from paths import DATA
import pytorch_lightning as pl
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch_geometric.datasets import ModelNet

# %%
''' Trying this out with the torch_geometric modules (DataLoader and ModelNet) '''

class ModelNetDataModule(pl.LightningDataModule):
    ''' PyTorch Lightning DataModule for ModelNet dataset '''
    def __init__(self, data_dir: str=DATA, sample_points: int=1024):
        super().__init__()
        self.data_dir = data_dir
        self.pre_transform = T.NormalizeScale()
        self.transform = T.SamplePoints(sample_points) # Need this to convert mesh into point cloud
        self.modelnet_dataset_alias = '10' # Specify 10 or 40 (ModelNet10, ModelNet40)
    
    def prepare_data(self) -> None:
        ModelNet(root=DATA,  train=True, name=self.modelnet_dataset_alias, pre_transform=self.pre_transform)
        ModelNet(root=DATA,  train=False, name=self.modelnet_dataset_alias, pre_transform=self.pre_transform)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":

            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [55000, 5000], generator=torch.Generator().manual_seed(42)
            )

        # Assign test dataset for use in dataloader(s)
        elif stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        elif stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=32)
