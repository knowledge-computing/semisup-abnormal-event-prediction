# semisup-abnormal-event-prediction
A Semi-Supervised Approach for Abnormal Event Prediction on Large Operational Network Time-Series Data

https://arxiv.org/abs/2110.07660

## Requirements
torch
seaborn
torch_geometric https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
E.g., pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.9.0+cu102.html

## How to Run
python train.py --data_path /data/yijun/NTT/graph-data/08/1NTT19213_naive_itv_72 --res_path /data/yijun/NTT/model-results-0111 --model_name TGConvNGATCluster2 --num_classes 2 --h_dim 32 --random_seed 1234 --lr 0.001 --eta 1 --use_unlabel --gpu_id 1 --batch_size 16

Average training one epoch: 6 minutes