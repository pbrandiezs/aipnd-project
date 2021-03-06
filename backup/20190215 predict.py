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
parser.add_argument('--gpu', action="store_true", dest='gpu', default=False, help='Use gpu for inference, default True')
args = parser.parse_args()
print(args)

print()
input_image = args.input
print("Input image is:", input_image)

checkpoint = args.checkpoint + "/checkpoint.pth"
print("Checkpoint location is", checkpoint)

gpu = args.gpu
print("GPU setting is:",gpu)

# Define settings
learning_rate=0.001
hidden_units=512

# Load previously saved checkpoint

# Loads checkpoint and rebuilds the model
def load_checkpoint(filepath):
#    if not gpu:
#       print("Not gpu")
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epochs = checkpoint['epochs']
    loss = checkpoint['loss']
    return model

model = models.vgg11(pretrained=True)
#criterion = nn.NLLLoss()
#optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)


# freeze
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

model = load_checkpoint(checkpoint)
model.eval()

print(model)