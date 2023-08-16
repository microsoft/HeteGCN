#!/bin/bash
python train.py --data_dir datasets --dataset mr --path FF-NF --wt_reg 10.0 --learning_rate 0.0005 --dropout 0.5 --emb_reg 0.0 --FF_norm None --NF_norm row --verbose