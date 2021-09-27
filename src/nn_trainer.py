# Code aspired from Tomas Beuzen
import json
import numpy as np
import pandas as pd
from collections import OrderedDict
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms, models, datasets
from torchsummary import summary
from IPython.display import Image
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset
from PIL import Image
import PIL.ImageOps
def trainer(model, criterion, optimizer, trainloader, validloader, epochs=5, verbose=True):
    train_loss = []
    valid_loss = []
    train_accuracy = []
    valid_accuracy = []
    for epoch in range(epochs):
        train_batch_loss = 0
        train_batch_acc = 0
        valid_batch_loss = 0
        valid_batch_acc = 0
        
        # Training
        for X, y in trainloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()       
            y_hat = model(X).flatten() 
            y_hat_labels = torch.sigmoid(y_hat) > 0.5        
            loss = criterion(y_hat, y.type(torch.float32))   
            loss.backward()             
            optimizer.step()            
            train_batch_loss += loss.item()  
            train_batch_acc += (y_hat_labels == y).type(torch.float32).mean().item()   
        train_loss.append(train_batch_loss / len(trainloader))     
        train_accuracy.append(train_batch_acc / len(trainloader)) 
        
        # Validation
        model.eval() 
        with torch.no_grad():  
            for X, y in validloader:
                X, y = X.to(device), y.to(device)
                y_hat = model(X).flatten()  
                y_hat_labels = torch.sigmoid(y_hat) > 0.5
                loss = criterion(y_hat, y.type(torch.float32))   
                valid_batch_loss += loss.item()                 
                valid_batch_acc += (y_hat_labels == y).type(torch.float32).mean().item()   
        valid_loss.append(valid_batch_loss / len(validloader))
        valid_accuracy.append(valid_batch_acc / len(validloader))  
        model.train()  
        
        # Print progress
        if verbose:
            print(f"Epoch {epoch + 1}:",
                  f"Train Accuracy: {train_accuracy[-1]:.2f}.",
                  f"Valid Accuracy: {valid_accuracy[-1]:.2f}.")

    
    results = {"train_accuracy": train_accuracy,
               "valid_accuracy": valid_accuracy}
    return results
