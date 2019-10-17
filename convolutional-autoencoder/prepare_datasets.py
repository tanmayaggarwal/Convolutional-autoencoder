import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# %matplotlib inline

# Prep the datasets

def prepare_datasets():
    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()

    # load the training and test datasets
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    # create training and test dataloaders
    num_workers = 0
    # number of samples per batch to load
    batch_size = 20

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, num_workers = num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, num_workers = num_workers)

    # visualize the data

    #obtain one batch of training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    images = images.numpy()

    # get one image from the batch
    img = np.squeeze(images[0])

    fig = plt.figure(figsize = (5,5))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')

    return (train_loader, test_loader, batch_size)
