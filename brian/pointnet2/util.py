# %%
from paths import DATA

import random
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T

# %% Plot 3D point cloud
def plot_3d_shape(shape):
    print('Number of data points: ', shape.pos.shape[0])
    x = shape.pos[:, 0]
    y = shape.pos[:, 1]
    z = shape.pos[:, 2]
    fig = px.scatter_3d(x=x, y=y, z=z, opacity=0.3)
    fig.update_traces(marker_size=3)
    fig.show()


# %% Visualize Confusion Matrix
def plot_confusion_matrix(conf_mat, classes, figsize=(5,5), text_size=10):
    fig, ax = plt.subplots(figsize=figsize)
    ax.matshow(conf_mat, cmap=plt.cm.Oranges, alpha=0.3)
    
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(x=j, y=i, s=conf_mat[i, j], fontsize=text_size, va='center', ha='center')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticklabels(classes, fontsize=12)
    plt.xticks(rotation=90)
        
    plt.show()


# %% Run in Interactive mode to see visualizations
if __name__=='__main__':
    dataset_val   = ModelNet(root             = DATA,
                            train            = False,
                            name             = '10',
                            pre_transform    = T.NormalizeScale(),
                            transform        = T.SamplePoints(1024))
    sample_idx = random.choice(range(len(dataset_val)))
    plot_3d_shape(dataset_val[sample_idx])

    dataset_val[sample_idx]

    classes = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
    conf_matrix = np.array([[22,  6,  2,  3,  1,  1,  0,  3,  7,  5],
       [ 6, 65,  3,  2,  0,  2,  1,  3,  8, 10],
       [ 1, 13, 55, 10,  0,  0,  1,  0, 11,  9],
       [ 0, 10,  2, 31,  0,  0,  2, 10, 19, 12],
       [ 8,  5,  0,  0, 15,  1,  4,  5,  0, 48],
       [ 3,  1,  1,  3,  1, 73,  0,  5,  0, 13],
       [ 3, 19,  2, 10,  8,  0,  6,  4,  2, 32],
       [16,  8,  0,  4,  1,  0,  0, 64,  3,  4],
       [ 0,  0,  0, 17,  0,  0,  1,  2, 80,  0],
       [ 7,  4,  0,  0,  4,  0,  0,  7,  0, 78]])

    plot_confusion_matrix(conf_matrix, classes)

