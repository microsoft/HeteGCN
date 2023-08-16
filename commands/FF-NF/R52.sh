#!/bin/bash
python train.py  --data_dir datasets --data_dir datasets --dataset R52 --path FF-NF --wt_reg 10.0 --learning_rate 0.002 --dropout 0.75 --emb_reg 0.0 --FF_norm None --NF_norm row --verbose