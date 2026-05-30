import numpy as np
import pandas as pd
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# VIT  
from torchvision import transforms
from torchvision.models.vision_transformer import VisionTransformer
from torchvision.models import vit_b_16, ViT_B_16_Weights

# Pillow for images 
from PIL import Image

# Test train split 
from sklearn.model_selection import train_test_split

# Progress bar 
from tqdm import tqdm

import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import time
import torch


# Defining image preprocessing pipeline 
def getImage(path, size, img_ID, bbox_dict, pad = 0.15):
    """
    Image preprocessing pipeline
    Loads an image from path, applies bounding box cropping with padding, and resizes the image for use in the ML models.
    
    

    Input:
    path (str): path to image.
    size (int): output image size.
    img_ID (int): image ID used to access boundary box information in bbox_dict.
    bbox_dict (dict): dictionary containing boundary box information.
    pad (float): % padding added around bounding box.

    Output: 
    image: PIL.Image.
    
    """
    # Open image 
    img = Image.open(path).convert("RGB")
    # Copy a local instance
    image = img.copy()
    # Close original image 
    img.close()

    # Get bounding box information
    x, y, w, h = bbox_dict[img_ID]

    # Get image dimensions
    img_w, img_h = image.size
    
    # Compute padding to add to boundary box (in pixels)
    pad_w = w * pad
    pad_h = h * pad

    # Define padded crop coordinates
    x1 = max(0, int(x - pad_w))
    y1 = max(0, int(y - pad_h))
    x2 = min(img_w, int(x + w + pad_w))
    y2 = min(img_h, int(y + h + pad_h))

    # Crop image using bounding box
    image = image.crop((x1, y1, x2, y2))

    # Resize cropped image
    image = image.resize((size, size))

    # Return processed image 
    return(image)


class CUBDataset(Dataset):

    def __init__(self, df, transform=None, size=224):
        """
        Initialise dataset.

        Input:
        df (pd.DataFrame): dataframe containing image information.
        transform: torchvision transform pipeline.
        size (int): output image size.
        """
        # --- Store inside class: ---
        self.df = df.reset_index(drop=True) # dataframe
        self.transform = transform          # transform pipeline
        self.size = size                    # output image size (used in getImage)

    def __len__(self):
        """
        Returns number of images in dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Loads and preprocesses a single image.

        Input:
        idx (int): image index.

        Output:
        image: processed image tensor.
        label: integer class label.
        """

        # Get df row corresponding to image index
        row = self.df.iloc[idx]

        # --- Load and preprocess image ---
        # (convert to RGB, crop to bounding box + padding, resize)
        image = getImage(
            path=row["Image Path"],
            size=self.size,
            img_ID=int(row["ID"]),
            bbox_dict=bbox_dict
        )

        # Apply transform pipeline (if transform is specified)
        if self.transform:
            image = self.transform(image)

        # Get class label (integer)
        label = label_map[row["Class"]]
        label = torch.tensor(label, dtype=torch.long)

        # Return image and label as a tuple
        return image, label