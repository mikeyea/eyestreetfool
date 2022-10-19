import numpy as np
import matplotlib.pyplot as plt

def process_image(image):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch       
    model, returns an Numpy array
    '''
    # Open the image
    from PIL import Image
    image = Image.open(image)
    # Resize the images where the shortest side is 256 pixels, keeping the aspect ratio
    if image.size[0] > image.size[1]:
        image.thumbnail((5000, 256))
    else:
        image.thumbnail((256, 5000))
    # Crop out the center 224x224 portion of the image
    left_margin = (image.width-224)/2
    bottom_margin = (image.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    image = image.crop((left_margin, bottom_margin, right_margin,   
                      top_margin))
    # Convert color channel values to float
    image = np.array(image)/255
   
    # Normalize
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225]) 
    image = (image - mean)/std
    
    # Reorder dimensions
    image = image.transpose((2, 0, 1))
    
    return image