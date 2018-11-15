## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        conv_dim = 32
        self.conv1 = nn.Conv2d(1, conv_dim, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(conv_dim, conv_dim*2, kernel_size=3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(conv_dim*2, conv_dim*4, kernel_size=3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(conv_dim*4, conv_dim*4, kernel_size=3, stride=1, padding=0)
        self.conv5 = nn.Conv2d(conv_dim*4, conv_dim*8, kernel_size=3, stride=1, padding=0)
        self.conv6 = nn.Conv2d(conv_dim*8, conv_dim*8, kernel_size=3, stride=1, padding=0)
        self.conv7 = nn.Conv2d(conv_dim*8, conv_dim*16, kernel_size=3, stride=1, padding=0)
        self.conv8 = nn.Conv2d(conv_dim*16, conv_dim*16, kernel_size=3, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(conv_dim, affine=True)
        self.bn2 = nn.BatchNorm2d(conv_dim*2, affine=True)
        self.bn3 = nn.BatchNorm2d(conv_dim*4, affine=True)
        self.bn4 = nn.BatchNorm2d(conv_dim*4, affine=True)
        self.bn5 = nn.BatchNorm2d(conv_dim*8, affine=True)
        self.bn6 = nn.BatchNorm2d(conv_dim*8, affine=True)
        self.bn7 = nn.BatchNorm2d(conv_dim*16, affine=True)
        self.bn8 = nn.BatchNorm2d(conv_dim*16, affine=True)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 136)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.5)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        


    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        # position encoding
        #b, c, h, w = x.shape
        x = self.relu(self.maxpool(self.bn1(self.conv1(x))))


        x = self.relu(self.maxpool(self.bn2(self.conv2(x))))


        x = self.relu(self.maxpool(self.bn3(self.conv3(x))))


        x = self.relu(self.bn4(self.conv4(x)))


        x = self.relu(self.maxpool(self.bn5(self.conv5(x))))


        x = self.relu(self.bn6(self.conv6(x)))


        x = self.relu(self.maxpool(self.bn7(self.conv7(x))))


        x = self.relu(self.bn8(self.conv8(x)))

        x = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], -1)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x
