#!/bin/bash
python train.py  --data_dir datasets --dataset 20ng --path FF-NF --wt_reg 100 --learning_rate 0.0005 --dropout 0.5 --emb_reg 0.0 --FF_norm None --NF_norm sym --verbose --cpu