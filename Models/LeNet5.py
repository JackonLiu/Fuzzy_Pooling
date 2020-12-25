import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """ Classical Lenet-5 structure as proposed by Yann LeCun
    Arguments:
        n_classes    : The number of classes in the data set, the output of the last softmax layer
    Returns:
        probs        : Probability of the each of the classes corresponding to the input batch
    Defaults:
        Activation   : ReLU for Convolutions and Fully Connected layers
                       Softmax for the output layer
        Initial Padding   : 0, since in the data transformations it has been made (32, 32)
    """

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()

        # Change the Pooling layers here to use the different custom ones
        # Make the padding zero in case Resize is used in the transformations
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)

        # A dummy input to implement the flatten layer, to get the first dimension
        x = torch.randn(28, 28).view(-1, 1, 28, 28)
        # This value is calculated only once per initialization of the model
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(in_features=self._to_linear, out_features=120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(in_features=84, out_features=n_classes)
        
    def convs(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        #x = F.relu(self.conv3(x))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = F.softmax(x, dim=1)
        return probs