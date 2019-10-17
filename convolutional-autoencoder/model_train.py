import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# %matplotlib inline

# training the model

def train(train_loader, model):
    # specify the loss function
    # using MSELoss given we are comparing pixel values in input and output images (i.e., regression) rather than probablistic values
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    # number of epochs to train the model
    n_epochs = 10

    for epoch in range(1, n_epochs+1):
        # monitor training loss
        train_loss = 0

        ###################
        # train the model #
        ###################
        for data in train_loader:
            # _ stands in for labels, here
            # no need to flatten images
            images, _ = data
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass
            outputs = model(images)
            # calculate the loss
            loss = criterion(outputs, images)
            # backward pass
            loss.backward()
            # perform a single optimization step
            optimizer.step()
            # update running training loss
            train_loss += loss.item()*images.size(0)

        # print avg training statistics
        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    return
