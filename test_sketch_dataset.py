#!/usr/bin/env python3
"""
Test script for SketchDataset to verify it works with the sketchy dataset.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils.sketch_dataset import SketchDataset

def test_sketch_dataset():
    """Test the SketchDataset with the sketchy dataset"""
    
    # Path to the sketchy dataset
    base_path = '/home/shared_data/Datasets/sketchy/train'
    
    print(f"Testing SketchDataset with base path: {base_path}")
    print(f"Checking if directories exist...")
    
    # Check if directories exist
    edge_dir = os.path.join(base_path, 'edges_inverted_bw')
    sketch_dir = os.path.join(base_path, 'sketch_bw')
    
    print(f"Edge directory: {edge_dir}")
    print(f"Sketch directory: {sketch_dir}")
    
    if not os.path.exists(edge_dir):
        print(f"ERROR: Edge directory does not exist: {edge_dir}")
        return False
    
    if not os.path.exists(sketch_dir):
        print(f"ERROR: Sketch directory does not exist: {sketch_dir}")
        return False
    
    # Count files in each directory
    edge_files = len([f for f in os.listdir(edge_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    sketch_files = len([f for f in os.listdir(sketch_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"Edge files: {edge_files}")
    print(f"Sketch files: {sketch_files}")
    
    # Create dataset
    try:
        dataset = SketchDataset(
            base_path=base_path,
            img_size=256,
            train=True,
            edge_dir='edges_inverted_bw',
            sketch_dir='sketch_bw'
        )
        
        print(f"Dataset created successfully!")
        print(f"Dataset size: {len(dataset)}")
        
        # Test getting a sample
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Edge image shape: {sample['image'].shape}")
        print(f"Sketch image shape: {sample['target'].shape}")
        print(f"Image ID: {sample['img_id']}")
        
        # Check if images are binary (0 or 1)
        edge_img = sample['image'].numpy()
        sketch_img = sample['target'].numpy()
        
        print(f"Edge image unique values: {np.unique(edge_img)}")
        print(f"Sketch image unique values: {np.unique(sketch_img)}")
        
        # Visualize a few samples
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for i in range(3):
            sample = dataset[i]
            
            # Edge image (input)
            edge_img = sample['image'].squeeze().numpy()
            axes[0, i].imshow(edge_img, cmap='gray')
            axes[0, i].set_title(f'Edge {i+1}')
            axes[0, i].axis('off')
            
            # Sketch image (target)
            sketch_img = sample['target'].squeeze().numpy()
            axes[1, i].imshow(sketch_img, cmap='gray')
            axes[1, i].set_title(f'Sketch {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('sketch_dataset_test.png', dpi=150, bbox_inches='tight')
        print("Test visualization saved as 'sketch_dataset_test.png'")
        
        return True
        
    except Exception as e:
        print(f"ERROR creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_sketch_dataset()
    if success:
        print("✅ SketchDataset test passed!")
    else:
        print("❌ SketchDataset test failed!")
        sys.exit(1)
