#!/bin/bash
python train.py  --data_dir datasets --dataset ohsumed --path FF-NF --wt_reg 0.0 --learning_rate 0.002 --dropout 0.0 --emb_reg 0.0 --FF_norm sym --NF_norm None --verbose