import os
import cv2
import torch
import logging
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from skimage.transform import resize

class SketchDataset(Dataset):
    """
    Dataset class for training sketch diffusion models.
    Loads binary edge images (edges_inverted_bw) as input and sketch images (sketch_bw) as target.
    Handles the hierarchical structure of the sketchy dataset.
    """
    
    def __init__(self, base_path: str, img_size: int = 256, train: bool = True, 
                 transforms=None, edge_dir: str = "edges_inverted_bw", 
                 sketch_dir: str = "sketch_bw"):
        """
        Constructor Method
        
        Inputs:
            - base_path: (String) Base directory containing edge and sketch folders
            - img_size: (Int) Image preprocessing resize, default=256
            - train: (Boolean) If True, uses train split, default=True
            - transforms: (object) Data augmentation transforms
            - edge_dir: (String) Directory name for edge images
            - sketch_dir: (String) Directory name for sketch images
        """
        super(SketchDataset, self).__init__()
        
        self.base_path = base_path
        self.img_size = (img_size, img_size) if img_size is not None else None
        self.transforms = transforms
        self.train = train
        self.edge_dir = edge_dir
        self.sketch_dir = sketch_dir
        
        # Paths to edge and sketch directories
        self.edge_path = os.path.join(base_path, edge_dir)
        self.sketch_path = os.path.join(base_path, sketch_dir)
        
        # Get list of image pairs
        self.image_pairs = self._get_image_pairs()
        
        # Ensure not empty
        assert len(self.image_pairs) > 0, 'Empty Dataset'
        
        # Log the dataset creation
        logging.info(f'Creating {"Train" if train else "Test"} sketch dataset with {len(self.image_pairs)} examples.')
    
    def _get_image_pairs(self):
        """Get list of image pairs that exist in both edge and sketch directories"""
        pairs = []
        
        # Get all categories from edge directory
        if not os.path.exists(self.edge_path):
            raise ValueError(f"Edge directory does not exist: {self.edge_path}")
        
        categories = [d for d in os.listdir(self.edge_path) 
                     if os.path.isdir(os.path.join(self.edge_path, d))]
        
        for category in categories:
            edge_cat_path = os.path.join(self.edge_path, category)
            sketch_cat_path = os.path.join(self.sketch_path, category)
            
            # Skip if sketch category doesn't exist
            if not os.path.exists(sketch_cat_path):
                continue
            
            # Get edge files in this category
            edge_files = [f for f in os.listdir(edge_cat_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for edge_file in edge_files:
                # Get base name without extension
                base_name = os.path.splitext(edge_file)[0]
                
                # Find corresponding sketch files (they have suffixes like -1, -2, etc.)
                sketch_files = [f for f in os.listdir(sketch_cat_path) 
                              if f.startswith(base_name + '-') and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                # Create pairs for each sketch file
                for sketch_file in sketch_files:
                    pairs.append({
                        'edge_path': os.path.join(edge_cat_path, edge_file),
                        'sketch_path': os.path.join(sketch_cat_path, sketch_file),
                        'category': category,
                        'edge_file': edge_file,
                        'sketch_file': sketch_file
                    })
        
        return pairs
    
    def __len__(self):
        return len(self.image_pairs)
    
    def preprocess(self, img_path, is_binary=True):
        """
        Preprocess image for training
        
        Inputs:
            - img_path: (String) Path to image file
            - is_binary: (Boolean) Whether image should be treated as binary
            
        Outputs:
            - img_tensor: (torch.Tensor) Preprocessed image tensor
        """
        # Load image
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        
        # Resize if needed
        if self.img_size:
            img = img.resize(self.img_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize to [0, 1] range
        if img_array.max() > 1:
            img_array = img_array / 255.0
        
        # For binary images, ensure they are truly binary (0 or 1)
        if is_binary:
            img_array = (img_array > 0.5).astype(np.float32)
        
        # Convert to tensor and add channel dimension
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)  # Add channel dimension
        
        return img_tensor
    
    def __getitem__(self, idx):
        """
        Get item from dataset
        
        Inputs:
            - idx: (Int) Index of item to retrieve
            
        Outputs:
            - target: (dict) Dictionary containing:
                - image: (torch.Tensor) Edge image (input)
                - target: (torch.Tensor) Sketch image (target)
                - img_id: (String) Image filename
                - img_path: (String) Image path
        """
        pair = self.image_pairs[idx]
        
        # Load edge image (input)
        edge_img = self.preprocess(pair['edge_path'], is_binary=True)
        
        # Load sketch image (target)
        sketch_img = self.preprocess(pair['sketch_path'], is_binary=True)
        
        # Apply transforms if available
        if self.transforms is not None:
            edge_img, sketch_img = self.transforms(edge_img, sketch_img)
        
        # Create target dictionary
        target = {
            'image': edge_img,      # Input: edge image
            'target': sketch_img,   # Target: sketch image
            'img_id': f"{pair['category']}_{pair['edge_file']}_{pair['sketch_file']}",
            'img_path': pair['sketch_path'],
            'img_size': self.img_size
        }
        
        return target 