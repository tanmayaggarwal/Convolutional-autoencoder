import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# %matplotlib inline

# Building the network

# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        self.conv1 = nn.Conv2d(1, 16, 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(16, 4, 3, stride = 1, padding = 1)
        self.maxpool = nn.MaxPool2d(2, stride = 2, padding = 0)

        ## decoder layers ##
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride = 2)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride = 2)

    def forward(self, x):
        ## encode ##
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)

        ## decode ##
        x = F.relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))

        return x
