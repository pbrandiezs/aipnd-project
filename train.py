
# Udacity AI Programming in Python Nanodegree - Final project
#
# Author: Perry Brandiezs
#
# train.py - this program will train the network.
#
# Usage: train.py -h
#


import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image


# Get command line arguments
parser = argparse.ArgumentParser(description='Train the network')
parser.add_argument('data_directory', help='data_directory')
parser.add_argument('--save_dir', '-s', type=str, dest='save_directory', help='Set directory to save checkpoints')
parser.add_argument('--arch', type=str, dest='architecture', default='vgg11', help='Set the architecture, default vgg11')
parser.add_argument('--learning_rate', type=float, dest='learning_rate', default=0.01, help='Set the learning rate, default 0.01')
parser.add_argument('--hidden_units', type=int, dest='hidden_units', default=512, help='Set the hidden units, default 512')
parser.add_argument('--epochs', type=int, dest='epochs', default=3, help='Set the number of epochs, default 3')
parser.add_argument('--gpu', action="store_true", dest='gpu', default=True, help='Use gpu for training, default True')
args = parser.parse_args()
print(args)

print("epochs", args.epochs)


model = "models." + args.architecture + "(pretrained=True)"

print("Model is:", model)
