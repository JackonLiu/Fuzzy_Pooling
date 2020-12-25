import numpy as np
from datetime import datetime
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms


if torch.cuda.is_available():
    device = torch.device("cuda:0")  # Can be continued like cuda:1 cuda:2....etc in case of multiple GPUs 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# Global Parameters
random_seed = 43
initial_learning_rate = 0.001
batch_size = 64
epochs = 25

img_size = 32
n_classes = 10

# Define transformations on the data set
#transforms = transforms.Compose([transforms.Resize((img_size, img_size)),
#                                transforms.ToTensor()])
#transforms = transforms.Compose([transforms.ToTensor(), 
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transforms = transforms.Compose([transforms.ToTensor()])

# Download the Data sets
train_dataset = datasets.CIFAR10(root='mnist_data', 
                               train=True, 
                               transform=transforms,
                               download=True)

valid_dataset = datasets.CIFAR10(root='mnist_data', 
                               train=False, 
                               transform=transforms)

print(valid_dataset)

# Data Loaders and iterators
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

# Shuffle does not matter in the case of validation and test sets
valid_loader = DataLoader(dataset=valid_dataset, 
                          batch_size=batch_size,
                          shuffle=False)


torch.manual_seed(random_seed)

model = LeNet5(n_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)
criterion = nn.CrossEntropyLoss()
model.apply(init_weights)

#optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device)