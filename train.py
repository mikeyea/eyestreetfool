#!/usr/bin/env python3
# -*- coding: utf-8 -*-                                                                         
# PROGRAMMER: Young-bai Yea
# DATE CREATED: October 13, 2022                                
# REVISED DATE: 
# PURPOSE: Trains a new network on a dataset and save the model as a checkpoint.
#
# Basic usage: python train.py data_directory
# Prints out training loss, validation loss, and validation accuracy as the network trains
#Options:
# Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
# Choose architecture: python train.py data_dir --arch "vgg13"
# Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
# Use GPU for training: python train.py data_dir --gpu
#    
##

# Imports python modules
from time import time, sleep

# Imports used by this program
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import json
from get_input_args import get_input_args '''import my files with functions'''
from load_data import load_data

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# Main program function defined below
def main():
    # Measures total program runtime by collecting start time
    start_time = time()
    
    # Get arguments data_dir, learning rate, hidden units, epochs
    in_arg = get_input_args()
    
    #Load data
    trainloader, validloader, testloader, class_to_idx = load_data(in_arg.data_dir)
    
    #Read classes and indices from JSON
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    # Set device variable to use GPU if available if not CPU
    def device(in_arg):
        if in_arg.gpu == 'y' and torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        return device

    # Import pre-trained model
    if in_arg.arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        model = models.vgg11(pretrained=True)
    
    # Freeze the feature parameters, so the model will not back-propagate through them
    
    for param in model.parameters():
        param.requires_grad = False

    # Define the model's classifier
    model.classifier = nn.Sequential(nn.Linear(25088, in_arg.hidden_units[0]),
                                     nn.ReLU(),
                                     nn.Dropout(in_arg.dropout),
                                     nn.Linear(in_arg.hidden_units[0], in_arg.hidden_units[1]),
                                     nn.ReLU(),
                                     nn.Dropout(in_arg.dropout),
                                     nn.Linear(in_arg.hidden_units[1], in_arg.hidden_units[2]),
                                     nn.ReLU(),
                                     nn.Dropout(in_arg.dropout),
                                     nn.Linear(in_arg.hidden_units[2], len(class_to_idx)),
                                     nn.LogSoftmax(dim=1))
    # Define criterion
    criterion = nn.NLLLoss()

    # Define optimizer (Adam or similar) and only update the classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.lr)

    # Move the model to available device
    model.to(device(in_arg))

    #Train the Model

    epochs = in_arg.ep
    steps = 0
    running_loss = 0
    print_every = 5

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
        
            #Move inputs and labels to the device
            inputs, labels = inputs.to(device(in_arg)), labels.to(device(in_arg))
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
    
            #Zero the gradient (since PyTorch “remembers” gradients)
            optimizer.zero_grad()
        
            #Back propagate
            loss.backward()
        
            #Step through optimizer to reset weights for layers and biases
            optimizer.step()
        
            #Track running loss in the batch
        
            running_loss += loss.item()
        
            #Validate the model
            #Write a if statement to validate using modular of ‘0’ (every 5 loops, step 5, 10, 15, …)
            if steps % print_every == 0:
            
                #Use model eval to test (turnoff dropout)
                model.eval()
                valid_loss = 0
                accuracy = 0
                #With, gradient turned off, write a loop to go through the test data using test loader
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device(in_arg)), labels.to(device(in_arg))
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        #Track test loss
                        valid_loss += batch_loss.item()
                    
                        #Calculate accuracy
                        ps = torch.exp(logps)
                        top_ps, top_class = ps.topk(1, dim=1)
                    
                        #Equality tensor is top class extracted above step, but labels have the same shape as top_class
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                         
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                      f"Valid accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
                         
    # Measure total program runtime by collecting end time
    end_time = time()
    
    # Computes overall runtime in seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time #calculate difference between end time and start time
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(int((tot_time%3600)%60)) )
    
    # Save checkpoint
    model.class_to_idx = class_to_idx
    model.idx_to_class = cat_to_name
    checkpoint = {'arch': 'vgg11',
                  'input_size': 25088,
                  'output_size': 102,
                  'hidden_layers': [in_arg.hidden_units[0], in_arg.hidden_units[1], in_arg.hidden_units[2]],
                  'state_dict': model.state_dict(),
                  'epochs': in_arg.ep,
                  'optimizer_state': optimizer.state_dict,
                  'dropout': in_arg.dropout,
                  'class_to_idx': model.class_to_idx,
                  'idx_to_class': model.idx_to_class}
    torch.save(checkpoint, in_arg.save_dir + 'checkpoint.pth')
    
# Call to main function to run the program
if __name__ == "__main__":
    main()