# %%
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_memlab import MemReporter
import socket

from paths import DATA
from model import PointNet2
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader # https://github.com/Lightning-AI/lightning/issues/1557 -> Using torch geometric's DataLoader should work..
from torch_geometric.datasets import ModelNet
# from torch_geometric.data.lightning import LightningDataset # PyG's support for PyTorch Lightning. May have to use this

# %%
class TrainPointNet2(pl.LightningModule):
    ''' Train PointNet++ using PyTorch Lightning to classify ModelNet dataset '''
    def __init__(self,
                 AUGMENTATIONS                  = T.SamplePoints(2048), # Need this to convert mesh into point cloud
                 LR                             = 0.001,
                 BATCH_SIZE                     = 64, # 8 if not subsampling, 128 if subsampling
                 N_EPOCHS                       = 30,
                 MODELNET_DATASET_ALIAS         = '10', # 'ModelNet10' or 'ModelNet40'
                 SET_ABSTRACTION_RATIO_1        = 0.748,
                 SET_ABSTRACTION_RADIUS_1       = 0.4817,
                 SET_ABSTRACTION_RATIO_2        = 0.3316,
                 SET_ABSTRACTION_RADIUS_2       = 0.2447,
                 DROPOUT                        = 0.1,
                 ):

        super(TrainPointNet2, self).__init__()

        self.save_hyperparameters()             # Need this later to load_from_checkpoint without providing the hyperparams again

        self.augmentations                      = AUGMENTATIONS
        self.lr                                 = LR
        self.bs                                 = BATCH_SIZE
        self.n_epochs                           = N_EPOCHS
        self.modelnet_dataset_alias             = MODELNET_DATASET_ALIAS

        self.set_abstraction_ratio_1            = SET_ABSTRACTION_RATIO_1
        self.set_abstraction_radius_1           = SET_ABSTRACTION_RADIUS_1
        self.set_abstraction_ratio_2            = SET_ABSTRACTION_RATIO_2
        self.set_abstraction_radius_2           = SET_ABSTRACTION_RADIUS_2
        self.dropout                            = DROPOUT


        self.model                  = PointNet2(set_abstraction_ratio_1   = self.set_abstraction_ratio_1, 
                                                set_abstraction_ratio_2   = self.set_abstraction_ratio_2,
                                                set_abstraction_radius_1  = self.set_abstraction_radius_1,
                                                set_abstraction_radius_2  = self.set_abstraction_radius_2,
                                                dropout                   = self.dropout, 
                                                n_classes                 = 10 if self.modelnet_dataset_alias=='10' else 40)

        # self.loss                               = F.nll_loss  # Functional form. The model itself returns log_softmax, so we use NLL Loss instead of nn.CrossEntropyLoss()
        self.loss                               = torch.nn.NLLLoss() # Class form. The model itself returns log_softmax, so we use NLL Loss instead of nn.CrossEntropyLoss()
        self.loss_cum                           = 0

        print('='*90)
        print('MODEL HYPERPARAMETERS')
        print('='*90)
        print(self.hparams)
        print('='*90)


    def setup(self, stage:str): # setup vs. prepare_data in multi-GPU setting: https://github.com/Lightning-AI/lightning/issues/2515#issuecomment-653943497
        # self.reporter = MemReporter(model) # Set up memory reporter
        # self.reporter.report()
        if stage == 'fit' or stage is None:
            start_time = datetime.now()
            print('Data Prep Starting time: {}'.format(start_time.strftime('%Y%m%d_%H%M%S')))
            modelnet_data_path = os.path.join(DATA, 'ModelNet{}'.format(self.modelnet_dataset_alias))

            # ModelNet(root=modelnet_data_path,  train=True, name=self.modelnet_dataset_alias, pre_transform=T.NormalizeScale()) # Specify 10 or 40 (ModelNet10, ModelNet40)

            self.dataset_train = ModelNet(
                    root             = modelnet_data_path,
                    train            = True,
                    name             = self.modelnet_dataset_alias,
                    pre_transform    = T.NormalizeScale(),
                    transform        = self.augmentations)

            self.dataset_val   = ModelNet(
                    root             = modelnet_data_path,
                    train            = False,
                    name             = self.modelnet_dataset_alias,
                    pre_transform    = T.NormalizeScale(),
                    transform        = T.SamplePoints(1024)) # Reduced augmentation for validation set
                    # transform        = self.augmentations)

            print('Ending time: {}'.format(datetime.now().strftime('%Y%m%d_%H%M%S')))
            print('Elapsed time: {}'.format(datetime.now()-start_time))
        
        # self.lightning_dataset = LightningDataset(dataset_train, dataset_val) # PyG's support for PyTorch Lightning
        
    def train_dataloader(self):
        train_dataloader = DataLoader(dataset        = self.dataset_train,
                                      batch_size     = self.bs,
                                      shuffle        = True,
                                      num_workers    = 8,
                                      pin_memory     = False) # pin_memory=True to keep the data in GPU
        return train_dataloader
        
        
    def val_dataloader(self):
        val_dataloader   = DataLoader(dataset        = self.dataset_val,
                                      batch_size     = self.bs,
                                      shuffle        = False,
                                      num_workers    = 8,
                                      pin_memory     = False) # pin_memory=True to keep the data in GPU
        return val_dataloader


    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                 step_size   = self.step_lr_step_size, # 20
    #                                                 gamma       = self.step_lr_gamma) # 0.5
    #     scheduler = {'scheduler': scheduler, 'interval': 'step'}
    #     return [optimizer], [scheduler]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        epochs              = self.n_epochs,
                                                        steps_per_epoch     = len(self.dataset_train)//self.bs, # The number of steps per epoch to train for. This is used along with epochs in order to infer the total number of steps in the cycle if a value for total_steps is not provided. Default: None
                                                        max_lr              = 0.0015,
                                                        pct_start           = 0.1,  # The percentage of the cycle spent increasing the learning rate Default: 0.3
                                                        div_factor          = 25,   # Determines the initial learning rate via initial_lr = max_lr/div_factor Default: 25
                                                        final_div_factor    = 1e3)  # Determines the minimum learning rate via min_lr = initial_lr/final_div_factor Default: 1e4
        scheduler = {'scheduler': scheduler, 'interval': 'step'}

        return [optimizer], [scheduler]


    def forward(self, data):
        return self.model(data)


    def training_step(self, batch, batch_idx):
        pred, target = self.forward(batch), batch.y
        loss = self.loss(pred, target)
        self.loss_cum += loss.item() # Not including the .item() will cause the loss to accumulate in GPU memory
        self.log('loss', loss)
        return {'loss': loss}


    def on_train_epoch_end(self):
        self.log('loss_epoch', self.loss_cum)
        self.loss_cum = 0
        # self.reporter.report() # Report memory usage
        return {'loss_epoch': self.loss_cum}
    

    def validation_step(self, batch, batch_idx):
        pred, target = self.forward(batch), batch.y
        loss = self.loss(pred, target)
        self.log('val_loss', loss)
        return {'val_loss': loss}


    # def test_step(self, batch, batch_idx):
    #     pred, target = self.forward(batch), batch.y
    #     loss = self.loss(pred, target)
    #     self.log('test_loss', loss)

# %%
if __name__=='__main__':
    torch.set_float32_matmul_precision('medium') # medium or high. See: https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    hostname = socket.gethostname()
    run_ID = '_'.join([hostname, datetime.now().strftime('%Y%m%d_%H%M%S')])
    print('Hostname: {}'.format(hostname))

    logger_tb = TensorBoardLogger('./tb_logs', name=run_ID)
    logger_wandb = WandbLogger(project='PointNet2', name=run_ID, mode='online') # online or disabled

    cb_checkpoint = ModelCheckpoint(dirpath     = './model_checkpoint/{}/'.format(run_ID),
                                    monitor     = 'val_loss',
                                    filename    = '{val_loss:.5f}-{loss:.5f}-{epoch:02d}',
                                    save_top_k  = 10)

    cb_lr_monitor = LearningRateMonitor(logging_interval='epoch')

    cb_earlystopping = EarlyStopping(monitor    = 'val_loss',
                                     patience   = 20,
                                     strict     = True, # whether to crash the training if monitor is not found in the val metrics
                                     verbose    = True,
                                     mode       = 'min')

    N_EPOCHS = 30
    # augmentations = T.Compose([
    #                     T.SamplePoints(2048),
    #                     T.RandomJitter(0.005),
    #                     T.RandomFlip(1),
    #                     T.RandomRotate(180),
    #                     T.RandomScale((0.8, 1.2)),
    #                     T.RandomShear(0.1)
    #                     ])
    augmentations = T.SamplePoints(2048)

    trainer = Trainer(
        max_epochs                      = N_EPOCHS,
        accelerator                     = 'gpu',  # set to cpu to address CUDA errors.
        strategy                        = 'ddp', # 'auto' or 'ddp' (other options probably available) # Currently only the pytorch_lightning.strategies.SingleDeviceStrategy and pytorch_lightning.strategies.DDPStrategy training strategies of PyTorch Lightning are supported in order to correctly share data across all devices/processes
        devices                         = 'auto',    # [0, 1] or use 'auto'
        log_every_n_steps               = 1,
        fast_dev_run                    = False,     # Run a single-batch through train & val and see if the code works
        logger                          = [logger_tb, logger_wandb],
        callbacks                       = [cb_checkpoint, cb_earlystopping, cb_lr_monitor])

    model = TrainPointNet2(
        N_EPOCHS                        = N_EPOCHS,
        AUGMENTATIONS                   = augmentations, # Need this to convert mesh into point cloud
    )

    trainer.fit(model)
