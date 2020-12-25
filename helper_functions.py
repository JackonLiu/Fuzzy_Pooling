import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")


def train(train_loader, model, criterion, optimizer, device):
    """Function for one single training loop pass through the entire dataset
    Arguments:
        train_loader : Data loader
        criterion    : Loss for the model
        device       : Device where the training loop executes, not where data is stored
    Returns :
        model       : Model with updated weights after training
        epoch_los   : Loss per entry of the training set for one epoch
    """

    #model.train()
    running_loss = 0
    
    for X, y_true in train_loader:

        optimizer.zero_grad()
        
        #X = X/255
        X = X.to(device)
        y_true = y_true.to(device)
    
        # Forward pass
        y_pred = model(X)
        loss = criterion(y_pred, y_true) 
        running_loss += loss.item() * X.size(0)

        loss.backward()
        optimizer.step()
        
    epoch_loss = running_loss / len(train_loader.dataset)
    return model, epoch_loss


def validate(valid_loader, model, criterion, device):
    """Function for the validation step of the training loop"""
   
    model.eval()
    running_loss = 0
    
    for X, y_true in valid_loader:
    
        #X = X/255
        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        y_hat = model(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
        
    return model, epoch_loss


def get_accuracy(model, data_loader, device):
    """Function for computing the accuracy of the predictions over the entire data_loader """
    
    correct_pred = 0 
    n = 0
    
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:

            #X = X/255
            X = X.to(device)
            y_true = y_true.to(device)

            y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n


def plot_losses(train_losses, valid_losses):
    """Function for plotting training and validation losses """
     
    plt.style.use('seaborn')

    train_losses = np.array(train_losses) 
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize = (8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss') 
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs", 
            xlabel='Epoch',
            ylabel='Loss') 
    ax.legend()
    fig.show()
    
    plt.style.use('default')


def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, print_every=1):
    """Function defining the entire training loop
    Arguments:
        model     : Model for training the Neural Network
        criterion : The loss metric for the loop
        optimizer : Updating the weights of the model
        train_loader, valid_loader    : Data loaders for train and validation sets respectively
        epochs    : Number of passes through the entire training loop
        print_every   : Frequency of printing of information such accuracies and losses

    Returns:
        model     : Model with updated weights after the entire training loops
        optimizer : Optimizer with the gradients
        (train_losses, valid_losses)  : Lists with the training and validation loss histories
    """
    
    train_losses = []
    valid_losses = []
    model.train()
 
    # Train model
    for epoch in range(0, epochs):

        model, train_loss = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)

        if epoch % print_every == (print_every - 1):
            
            train_acc = get_accuracy(model, train_loader, device=device)
            valid_acc = get_accuracy(model, valid_loader, device=device)
                
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

    plot_losses(train_losses, valid_losses)
    
    return model, optimizer, (train_losses, valid_losses)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def multi_plot(model, rows=5, cols=10):
    fig = plt.figure()
    for index in range(1, rows*cols + 1):
        plt.subplot(rows, cols, index)
        plt.axis('off')
        plt.imshow(valid_dataset.data[index], cmap='gray_r')
    
        with torch.no_grad():
            model.eval()
            probs = model(valid_dataset[index][0].unsqueeze(0))
        
        title = f'{torch.argmax(probs)} ({torch.max(probs * 100):.0f}%)'
    
        plt.title(title, fontsize=7)
    fig.suptitle('LeNet-5 - predictions')

