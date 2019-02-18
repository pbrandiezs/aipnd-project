# Udacity AI Programming in Python Nanodegree - Final project
#
# Author: Perry Brandiezs
#
# predict.py - this program will be used to predict a classification from an image
#
# Usage: predict.py -h
#


import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json



# Get command line arguments
parser = argparse.ArgumentParser(description='Use a trained network to predict the class for an input image')
parser.add_argument('input', type=str, help='Path to input image')
parser.add_argument('checkpoint', type=str, help='Path to checkpoint save directory')
parser.add_argument('--top_k', type=int, dest='top_k', default=3, help='top_k values to display, default 3')
parser.add_argument('--gpu', action="store_true", dest='gpu', default=True, help='Use gpu for inference, default True')
args = parser.parse_args()
print(args)

print()
input_image = args.input
print("Input image is:", input_image)

checkpoint = args.checkpoint + "/checkpoint"
print("Checkpoint location is", checkpoint)

gpu = args.gpu
print("GPU setting is:",gpu)