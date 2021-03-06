
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
import json


# Get command line arguments
parser = argparse.ArgumentParser(description='Train the network')
parser.add_argument('data_directory', type=str, help='data_directory')
parser.add_argument('--save_dir', '-s', type=str, dest='save_directory', help='Set directory to save checkpoints')
parser.add_argument('--arch', dest='architecture', default='vgg11', help='Set the architecture, default vgg11')
parser.add_argument('--learning_rate', type=float, dest='learning_rate', default=0.01, help='Set the learning rate, default 0.01')
parser.add_argument('--hidden_units', type=int, dest='hidden_units', default=512, help='Set the hidden units, default 512')
parser.add_argument('--epochs', type=int, dest='epochs', default=3, help='Set the number of epochs, default 3')
parser.add_argument('--gpu', action="store_true", dest='gpu', default=True, help='Use gpu for training, default True')
args = parser.parse_args()
print(args)

# print("epochs", args.epochs)

# set the model to use
# model = "models." + args.architecture + "(pretrained=True)"
model = models.__dict__[args.architecture](pretrained=True)

# set the data_dir
data_dir = args.data_directory
print("Data Directory is:", data_dir)

# set the train_dir, valid_dir, and test_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
print("train_dir:", train_dir)
print("valid_dir:", valid_dir)
print("test_dir:", test_dir)


# set the transforms
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
                                                           
data_transforms = {'train':train_transforms, 'test':test_transforms, 'validation':validation_transforms}

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)

image_datasets = {'train':train_data, 'test':test_data, 'validation':validation_data}

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
validationloader = torch.utils.data.DataLoader(validation_data, batch_size=32)

dataloaders = {'train':trainloader, 'test':testloader, 'validation':validationloader}

# Get the names
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# Build and train your network

# freeze
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 500)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(500, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier
print(model)

#Train
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
epochs = 3
print_every = 40
steps = 0

# change to cuda
# model.to('cuda')

for e in range(epochs):
    running_loss = 0
    for ii, (inputs, labels) in enumerate(trainloader):
        steps += 1
        
        #inputs, labels = inputs.to('cuda'), labels.to('cuda')
        inputs, labels = inputs.to('cpu'), labels.to('cpu')

        
        optimizer.zero_grad()
        
        # Forward and backward passes
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0:
            print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Loss: {:.4f}".format(running_loss/print_every))
            
            running_loss = 0

