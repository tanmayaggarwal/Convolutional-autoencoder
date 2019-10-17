# Convolutional-autoencoder

This repository contains a set of python files that define and train a convolutional autoencoder

Context:

Convolutional autoencoders improve the performance of an autoencoder relative to a linear autoencoder.
This repo builds a convolutional autoencoder to compress the MNIST dataset.

The encoder portion is made of 2 convolutional layers, each of which are followed by a max pooling layer.
The decoder portion is made of 2 transpose convolutional layers that learn to "upsample" a compressed representation.

Each of the hidden layers go through a ReLu activation, except for the output layer which goes through a sigmoid activation.

The original images have size 28x28 = 784
The final encoded layer has size 7x7x4 = 196 (25% the size of the original image)


The following dimensions are used for each of the layers:

## encoder layers ##
        self.conv1 = nn.Conv2d(1, 16, 3, stride = 1, padding = 1) # size: 28x28x16 Convolution
        self.conv2 = nn.Conv2d(16, 4, 3, stride = 1, padding = 1) # size: 14x14x4 Convolution
        self.maxpool = nn.MaxPool2d(2, stride = 2, padding = 0) # Reduces h / w by a factor of 2

## decoder layers ##
        ## a kernel of 2 and a stride of 2 increase the spatial dims by 2
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride = 2) # size: 14x14x16 Convolution
        self.t_conv2 = nn.ConvTranspose2d(16, 1, 2, stride = 2) # size: 28x28x1 Convolution
        
The model is trained for 10 epochs.

Notes:
- Upsampling (combined with normal convolution layers) can also be used as an alternative approach to decoding the compressed representation.
