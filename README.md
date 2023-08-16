This repository contains the source code for HeteGCN: Heterogeneous Graph Convolutional Networks for Text Classification.

This code supports constructing arbitraty HeteGCN paths consisting of layers [FF, NF, FN]. The layers should be compatible. 
Example HeteGCN Paths
    1. FF-NF
    2. NF-FN-FN
    3. FN-NF


To Run
- Example Command
    python train.py  --data_dir datasets --dataset mr --path NF-FN-NF --wt_reg 10.0 --learning_rate 0.002 --dropout 0.25 --emb_reg 0.0 --FN_norm None --NF_norm sym --verbose