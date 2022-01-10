# semisup-abnormal-event-prediction
A Semi-Supervised Approach for Abnormal Event Prediction on Large Operational Network Time-Series Data

https://arxiv.org/abs/2110.07660

## Requirements
* torch==1.9
* seaborn
* torch_geometric (check out https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) <br>
  Example: pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cu102.html


## Data
In this project, we are looking at network data. The below image shows an example network structure.

<p align="center">
  <img src="/images/network_deff.jpg" width="500">
</p>

In a network, there can be multiple devices. In this example network, we can see there are two devices presented in blue squares. One device usually has multiple descriptions or interfaces, like SDWAN device usually has lan and wan interfaces. Each device can have different types of interfaces. One description is measured by several attributes, and each attribute provides measurements as a time series. 

Besides the network data, we also have event data, which are reported at the device-level. The below image shows the process of labeling events. Here, we only have the reported events, and the system status at the remaining time are not explicitly provided, that means we are not sure about how long the event will be last for. So there are some timestamps near event time remain unlabeled in the dataset.

<p align="center">
  <img src="/images/label_construction.jpg" width="400">
</p>


## Running the code

__You can find more detail description by using "python train.py -h"__

>Some commonly-used paramters:
>- [--data_path]: parent path to the dataset, there must be an original data file named as "graph.npz" in the path
>- [--res_path]: parent path to the results, under which there will be /models, /logs, and /imgs
>- [--model_name]: model name, e.g., TGConvNGATCluster
>- [--num_classes]: number of classes in the output, default is 2
>- [--gpu_id]: GPU ID

Example command: python train.py --data_path /data/yijun/NTT/graph-data/08/1NTT19213_naive_itv_72 --res_path /data/yijun/NTT/model-results-0111 --model_name TGConvNGATCluster2 --num_classes 2 --h_dim 32 --random_seed 1234 --lr 0.001 --eta 1 --use_unlabel --gpu_id 1 --batch_size 16

## Visualizing results

__You can find the current saved model settings in [RES_PATH]/output.json__

__You can find the training logs in [RES_PATH]/logs/__

__You can find the intermediate visualization results in [RES_PATH]/imgs/__

The following images shows two examples of the clusters for __event__ (0) and __non-event__ (1) in the embedding space

![res1](/images/res1.jpg)
![res1](/images/res2.jpg)
