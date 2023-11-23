# %%
import os
import torch
import random

from paths import DATA
from train import TrainPointNet2

import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ModelNet
from torch_geometric.data import Data, Batch

from tqdm import tqdm
from util import plot_3d_shape, plot_confusion_matrix
from constants import CLASSES_MODELNET_10, CLASSES_MODELNET_40

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# %%
CKPT_PATH = '/home/brian/github/DL_practice/PointNet2/model_checkpoint/aorus_20231119_133124'
CKPT_FILENAME = 'val_loss=0.36129-loss=0.45817-epoch=28.ckpt'
MODELNET_DATASET_ALIAS = '10'

# %%
trainer = TrainPointNet2.load_from_checkpoint(os.path.join(CKPT_PATH, CKPT_FILENAME), map_location=torch.device('cpu'))

modelnet_data_path = os.path.join(DATA, 'ModelNet{}'.format(MODELNET_DATASET_ALIAS))
dataset_val   = ModelNet(root             = modelnet_data_path,
                         train            = False,
                         name             = MODELNET_DATASET_ALIAS,
                         pre_transform    = T.NormalizeScale(),
                         transform        = T.SamplePoints(1024))


val_dataloader = DataLoader(dataset        = dataset_val,
                            batch_size     = 128,
                            shuffle        = True,
                            num_workers    = 8,
                            pin_memory     = False)


# %% Plot a random 3D point cloud and print pred & actual
random_index = random.choice(range(len(dataset_val)))
print('='*90)
plot_3d_shape(dataset_val[random_index])

single_batch = Batch.from_data_list([dataset_val[random_index]])
out = trainer(single_batch)
pred = torch.argmax(out, dim=1)
actual = single_batch.y

class_names = CLASSES_MODELNET_10 if MODELNET_DATASET_ALIAS == '10' else CLASSES_MODELNET_40
print('Predicted:\t', class_names[pred.item()])
print('Actual:\t\t', class_names[actual.item()])


# %% Calculate classification metrics
targets, preds = [], []
for batch in tqdm(val_dataloader):
    actual = batch.y
    pred = torch.argmax(trainer(batch), dim=1)
    targets.append(actual)
    preds.append(pred)

targets = torch.cat(targets)
preds = torch.cat(preds)

# %%
y_true = targets.numpy()
y_pred = preds.numpy()

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')  
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

conf_matrix = confusion_matrix(y_true, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# print("Confusion Matrix:\n", conf_matrix)
figsize = (5, 5) if MODELNET_DATASET_ALIAS == '10' else (15, 15)
plot_confusion_matrix(conf_matrix, classes=class_names, figsize=figsize, text_size=10)

# %%
