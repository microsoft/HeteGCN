import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import warnings
warnings.filterwarnings("ignore")

import pickle 
import argparse
import pandas as pd
import tensorflow as tf
 
from model import Model
from utils import set_seed
from dataset import Dataset 

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str)
parser.add_argument("--mount_dir", default='.', type=str)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--output_dir", default='outputs', type=str)
parser.add_argument("--extended_data_dir", type=str, default=None)
parser.add_argument('--verbose', dest='verbose', action='store_true')
parser.add_argument('--cpu', dest='cpu', action='store_true')
parser.add_argument('--premultiply', dest='premultiply', action='store_true')

parser.add_argument("--path", default='NF', type=str)

parser.add_argument("--seed", default=2, type=int)
parser.add_argument("--hidden_dims", default=200, type=int)
parser.add_argument("--wt_reg", default=100, type=float)
parser.add_argument("--emb_reg", default=0, type=float)
parser.add_argument("--dropout", default=0.5, type=float)
parser.add_argument("--learning_rate", default=2e-3, type=float)
parser.add_argument("--decay_rate", default=0.99, type=float)
parser.add_argument("--decay_freq", default=50, type=int)
parser.add_argument("--epochs", default=300, type=int)
parser.add_argument("--early_stopping", default=30, type=int)
parser.add_argument("--batch_size_train", default=100000, type=int)

parser.add_argument("--NF_norm", default='None', choices=['row','sym','None'], type=str)
parser.add_argument("--FN_norm", default='None', choices=['row','sym','None'], type=str)
parser.add_argument("--NN_norm", default='None', choices=['row','sym','None'], type=str)
parser.add_argument("--FF_norm", default='None', choices=['row','sym','None'], type=str)
parser.add_argument("--table", default=None, type=str)

parser.add_argument("--use_feature_embeddings", default=0, type=int)
parser.add_argument("--feature_embeddings", type=str)

parser_args = parser.parse_args()

if parser_args.cpu:
    # To Train models with path containing FF/NN on CPU
    # Smaller datasets can still be trained on GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Prepare Dataset for consumption
dataset = Dataset(vars(parser_args))
args = dataset.get_args()

# Set Seeds for reproducability
set_seed(args['seed'])

# Create Model
model = Model(args.copy())

# Train The Model
sess = tf.Session()
sess.run(tf.global_variables_initializer())
results, aux_outputs = model.fit(sess, args)

print('\n--------------------------------------------')
print('Final Results')
print('--------------------------------------------')
print('Train Accuracy - ', results['TrainAccuracy'])
print('Train Micro F1 - ', results['TrainMicro'])
print('Train Macro F1 - ', results['TrainMacro'])
print()
print('Val Accuracy - ', results['ValAccuracy'])
print('Val Micro F1 - ', results['ValMicro'])
print('Val Macro F1 - ', results['ValMacro'])
print()
print('Test Accuracy - ', results['TestAccuracy'])
print('Test Micro F1 - ', results['TestMicro'])
print('Test Macro F1 - ', results['TestMacro'])
print('--------------------------------------------')