#!/usr/bin/env python3
# -*- coding: utf-8 -*-                                                                         
# PROGRAMMER: Young-bai Yea
# DATE CREATED: October 15, 2022                                
# REVISED DATE: 
# PURPOSE: Test a neural network and prints out inference output
#
# Predict flower name from an image with predict.py along with the probability of that name. 
#That is, you'll pass in a single image /path/to/image and return the flower name and class probability.
#   Return top KK most likely classes: python predict.py input checkpoint --top_k 3
#   Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
#   Use GPU for inference: python predict.py input checkpoint --gpu
#   Arguments: image path: 'flowers/test/95/image_07519.jpg' checkpoint path: 'checkpoint.pth'
##

# Imports python modules

# Imports used by this program
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import json
from get_input_args_pred import get_input_args_pred
from process_image import process_image
from PIL import Image

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# Main program function defined below
def main():
    
    # Get arguments image_path, model_path, top_k classes, cat_to_name mapping, and device setting  
    in_arg = get_input_args_pred()
    
    #Read classes and indices from JSON
    with open(in_arg.cat_names, 'r') as f:
        cat_to_name = json.load(f)
        
    # Set device variable to use GPU if available if not CPU
    def device(in_arg):
        if in_arg.gpu == 'y' and torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        return device

    # TODO: Write a function that loads a checkpoint and rebuilds the model
    def load_checkpoint(in_arg):
        checkpoint = torch.load(in_arg.model_path)
        model = checkpoint['arch']
        if checkpoint['arch'] == 'vgg11':
            model = models.vgg11(pretrained=True)
        else:
            model = models.vgg13(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.class_to_idx = checkpoint['class_to_idx']
        model.idx_to_class = checkpoint['idx_to_class']
        model.classifier = nn.Sequential(nn.Linear(25088, checkpoint['hidden_layers'][0]),
                                         nn.ReLU(),
                                         nn.Dropout(.2),
                                         nn.Linear(checkpoint['hidden_layers'][0], checkpoint['hidden_layers'][1]),
                                         nn.ReLU(),
                                         nn.Dropout(.2),
                                         nn.Linear(checkpoint['hidden_layers'][1], checkpoint['hidden_layers'][2]),
                                         nn.ReLU(),
                                         nn.Dropout(.2),
                                         nn.Linear(checkpoint['hidden_layers'][2], len(checkpoint['idx_to_class'])),
                                         nn.LogSoftmax(dim=1))
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device(in_arg))
        return model
    
    model = load_checkpoint(in_arg)

    def predict(image_path, model, top_val):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''  
        # Process image
        image = process_image(image_path)
    
        # Convert a numpy array to a tensor
        image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    
        # Need a list of 1 since only 1 image is going through
        image_input = image_tensor.unsqueeze(0)
        image_input = image_input.to(device(in_arg))
        
        # Probs from running inference
        probs = torch.exp(model.forward(image_input))
    
        # Top probs and classes from top K largest values
        top_probs, top_classes = probs.topk(top_val)
   
        # Move cuda tensors to cpu
        top_probs = top_probs.cpu()
        top_classes = top_classes.cpu()
    
        # Convert tensors to numpy arrays and make it hashable
        top_probs = top_probs.detach().numpy().tolist()[0] 
        top_classes = top_classes.detach().numpy().tolist()[0]
    
        # Convert indices to classes
        idx_to_class = {val: key for key, val in    
                             model.class_to_idx.items()}
        top_indices = [idx_to_class[lab] for lab in top_classes]
    
        # Use the provided JSON object read in earlier
        top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_classes]
        
        return top_probs, top_indices, top_flowers
  
    top_probs, top_indices, top_flowers = predict(in_arg.image_path, model, in_arg.top_k) 
    
    print('Top Flowers: ', top_flowers)
    print('Top Probabilities: ', top_probs)
    
# Call to main function to run the program
if __name__ == "__main__":
    main()