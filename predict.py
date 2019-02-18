# Udacity AI Programming in Python Nanodegree - Final project
#
# Author: Perry Brandiezs
#
# predict.py - this program will be used to predict a classification from an image
#
# Usage: predict.py -h
#

import matplotlib.pyplot as plt
import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json
import seaborn as sns



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

# get the names
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


    
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




#Process the test image

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    #open the image
    im = Image.open(image)

    #resize the image
    im = im.resize((255,255))
    
    #crop the image
    left = 16
    right = 240
    top = 16
    bottom = 240
    im = im.crop((left, top, right, bottom))
    np_image = np.array(im)
    
    # display the image being worked on.
    #flower_num = image.split('/')[2]
    #plt.imshow(np_image)
    #plt.title(cat_to_name[flower_num])
    
    # scale to between 0 and 1
    np_image = np_image / 255
    
    mean = np.array([0.485, 0.456, 0.406])
    std_dev = np.array([0.229, 0.224, 0.225])
    
    np_image = (np_image - mean) / std_dev
    
    #transpose 
    np_image = np_image.transpose((2, 0, 1))
    
    #print the shape to verify
    # print(np_image.shape)
    
    #return the converted image, ready for PyTorch
    return np_image
    
process_image(input_image)




def imshow(image, ax=None, title=None):
    #if ax is None:
    #    fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    # ax.imshow(image)
    
    return ax

# test it out
imshow(process_image("flowers/test/1/image_06743.jpg"))


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    proc_image = process_image(image_path)
    
    img = torch.from_numpy(proc_image).type(torch.FloatTensor)
    img = img.unsqueeze(0)
    
    # print(img.shape)
    
    probs = torch.exp(model.forward(img))
    # print("probs is", probs)
    #print("probs.topk is ", probs.topk(topk))
    top_probs, top_labs = probs.topk(topk)
    
    # print("top_probs", top_probs)
    #print("top_labs ", top_labs)
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_labs = top_labs.detach().numpy().tolist()[0]
    # print("top_probs detached is ", top_probs)
    # print("top_labs detached is ", top_labs)
    
    #index to classes
    name_to_cat = {val: key for key, val in
                   cat_to_name.items()}
    
    top_labels = []
    top_flowers = []
    # print("top_labs", top_labs)
    # print("cat_to_name", cat_to_name)
    #print("name for 21 ", cat_to_name['21'])
    # for lab in top_labs:
    #    print("name_to_cat of lab ", lab, "is ", cat_to_name[str(lab + 1)])
    top_labels = [cat_to_name[str(lab + 1)] for lab in top_labs]
    # top_flowers = [cat_to_name[lab] for lab in top_labs]
    top_classes = [str(lab + 1) for lab in top_labs]
    # print("Classes: ", top_classes)
    # return top_probs, top_labels, top_flowers
    return top_probs, top_labels, top_classes

#test

image_path = "flowers/test/1/image_06743.jpg"
probs, labels, classes = predict(image_path, model)
print(probs)
print(labels)
print(classes)






# TODO: Display an image along with the top 5 classes

#open the image
image = "flowers/test/1/image_06743.jpg"
im = Image.open(image)

# display the image being worked on.
flower_num = image.split('/')[2]
np_image = np.array(im)


# fig = plt.figure(figsize = (6,10))
# plt.subplot(2,1,1)
#plt.imshow(np_image)
#plt.title(cat_to_name[flower_num])
print("Flower name:", cat_to_name[flower_num])

#display the results
# probabilities, labels, flowers = predict(image, model)
probabilities, labels, classes = predict(image, model)
#plt.subplot(2,1,2)

#sns.barplot(x = probabilities, y = labels, color = sns.color_palette()[0])
#plt.show()
