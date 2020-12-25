import torch
import torch.nn as nn
import torch.nn.functional as F
from other_pooling import *
from Fuzzy_type2 import *

#Note: Some of the values written here are for the CIFAR10 dataset

class AlexNet(nn.Module):
    """ Classical Alexnet(2012) structure; Values written for 32x32 images
    Arguments:
        n_classes    : The number of classes in the data set, the output of the last softmax layer
    Returns:
        probs        : Probability of the each of the classes corresponding to the input batch
    Defaults:
        Activation   : ReLU for Convolutions and Fully Connected layers
                       Softmax for the output layer
        Dropout      : Default Pytorch Value of 0.5
    """

    def __init__(self, n_classes=10):
        super(AlexNet, self).__init__()
        # Change the Pooling layers here to use the different custom ones
        # Change the initial padding in case something like MNIST dataset is used

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = Tree_level2(kernel_size=2, stride=2)

        # A dummy input to implement the flatten layer, to get the first dimension
        # This value is calculated only once per initialization of the model
        x = torch.randn(3, 32, 32).view(-1, 3, 32, 32)
        self._to_linear = None
        self.features(x)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=self._to_linear, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, n_classes),
        )

    def features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self._to_linear)
        x = self.classifier(x)
        #x = F.softmax(x, dim=1)
        return x