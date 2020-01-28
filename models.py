## Define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## Define all the layers of this CNN, the requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## The last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.conv4 = nn.Conv2d(128, 256, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(0.1)
        self.dropout2 = nn.Dropout2d(0.2)
        self.dropout3 = nn.Dropout2d(0.3)
        self.dropout4 = nn.Dropout2d(0.4)
        self.dropout5 = nn.Dropout2d(0.5)
        self.dropout6 = nn.Dropout2d(0.6)
        self.fc1 = nn.Linear(43264,1000)
        self.fc2 = nn.Linear(1000,1000)
        self.fc3 = nn.Linear(1000,136)
        
        
    def forward(self, x):
        ## Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.dropout3(x)
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.dropout4(x)
        x = x.view(x.size(0), -1)  # also for flatten x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = self.dropout5(x)
        x = F.relu(self.fc2(x))
        x = self.dropout6(x)
        x = self.fc3(x)
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

