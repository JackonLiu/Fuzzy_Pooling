import numpy as np
from datetime import datetime
import time
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from Models import AlexNet
from helper_functions import *


parser = argparse.ArgumentParser(description="Multiple Pooling Methods")
parser.add_argument('--lr', default=0.001, type=float, help='initial learning_rate')
parser.add_argument('--epoch', default=5, type=int, help='number of epochs')
parser.add_argument('--batch_size', default=64, type=int, help='train and test batch size')
parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool, help='whether to use GPU')

args = parser.parse_args()

initial_learning_rate = args.lr
epochs = args.epoch
batch_size = args.batch_size
cuda = args.cuda


if cuda:
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# Global Parameters
random_seed = 43
img_size = 32
n_classes = 10

# Define transformations on the data set
#transforms = transforms.Compose([transforms.Resize((img_size, img_size)),
#                                transforms.ToTensor()])
#transforms = transforms.Compose([transforms.ToTensor(), 
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transforms = transforms.Compose([transforms.ToTensor()])

# Download the Data sets
train_dataset = datasets.CIFAR10(root='cifar10_data', 
                               train=True, 
                               transform=transforms,
                               download=True)

valid_dataset = datasets.CIFAR10(root='cifar10_data', 
                               train=False, 
                               transform=transforms)

print(train_dataset)

# Data Loaders and iterators
train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

# Shuffle does not matter in the case of validation and test sets
valid_loader = DataLoader(dataset=valid_dataset, 
                          batch_size=batch_size,
                          shuffle=False)


torch.manual_seed(random_seed)

model = AlexNet.AlexNet(n_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate)
criterion = nn.CrossEntropyLoss()
model.apply(init_weights)

model, optimizer, (train_losses, valid_losses) = training_loop(model, criterion, optimizer, train_loader, valid_loader, 5, device)
out_path = "saved_models"+randint(1000)
torch.save(model, out_path)
print("Checkpoint saved to {}".format(out_path))
plot_losses(train_losses, valid_losses)