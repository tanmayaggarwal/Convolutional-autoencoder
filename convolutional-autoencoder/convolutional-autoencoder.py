import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# %matplotlib inline

# Prep the datasets
from prepare_datasets import prepare_datasets
train_loader, test_loader, batch_size = prepare_datasets()

# build the network architecture

from model_arch import ConvAutoencoder
model = ConvAutoencoder()

# train the network

from model_train import train
train(train_loader, model)

# check our results

from check_results import check_results
check_results(test_loader, model, batch_size)

## END ##
