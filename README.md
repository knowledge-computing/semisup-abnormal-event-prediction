# semisup-abnormal-event-prediction
A Semi-Supervised Approach for Abnormal Event Prediction on Large Operational Network Time-Series Data

https://arxiv.org/abs/2110.07660

The image shows an example of the network structure ![network structure](/images/network_deff.jpg)

## Requirements
* torch==1.9
* seaborn
* torch_geometric (check out https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) <br>
  Example: pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cu102.html

## Running the code

__You can find a more detail in "*python train.py -h"__

>Some commonly-used paramters:
>- [--data_path]: parent path to the dataset, there must be an original data file named as "graph.npz" in the path
>- [--res_path]: parent path to the results, under which there will be /models, /logs, and /imgs
>- [--model_name]: model name, e.g., TGConvNGATCluster
>- [--num_classes]: number of classes in the output, default is 2
>- [--gpu_id]: GPU ID

>- Example command: python train.py --data_path /data/yijun/NTT/graph-data/08/1NTT19213_naive_itv_72 --res_path /data/yijun/NTT/model-results-0111 --model_name TGConvNGATCluster2 --num_classes 2 --h_dim 32 --random_seed 1234 --lr 0.001 --eta 1 --use_unlabel --gpu_id 1 --batch_size 16

## Interpreting / Visualizing results

__You can find the current saved model settings in [RES_PATH]/output.json__

__You can find the intermediate visualization results in [RES_PATH]/imgs/__

The following images shows two examples of the embedding clusters for __event__ (0) and __non-event__ (1) groups after training

![res1](/images/res1.jpg)
![res1](/images/res2.jpg)
