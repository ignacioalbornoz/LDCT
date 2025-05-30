import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path

class QuickDrawDataset(Dataset):
    def __init__(self, 
                 top3000_dir: str,
                 quickdraw_jpg_dir: str,
                 img_size: int = 64,
                 train: bool = True):
        """
        Args:
            top3000_dir (str): Path to directory containing top3000 txt files
            quickdraw_jpg_dir (str): Path to directory containing jpg images
            img_size (int): Size to resize images to
            train (bool): Whether this is training set
        """
        self.img_size = img_size
        self.train = train
        
        # Verificar directorios
        if not os.path.exists(top3000_dir):
            raise ValueError(f"Directory not found: {top3000_dir}")
        if not os.path.exists(quickdraw_jpg_dir):
            raise ValueError(f"Directory not found: {quickdraw_jpg_dir}")
            
        # Obtener lista de archivos top3000 (ignorar worst100)
        self.top3000_files = [f for f in os.listdir(top3000_dir) 
                             if f.endswith('_top3000.txt') and not f.endswith('_worst100.txt')]
        
        print(f"Found {len(self.top3000_files)} top3000 files")
        
        # Cargar todos los IDs de imágenes de los archivos top3000
        self.image_paths = []
        for top3000_file in self.top3000_files:
            category = top3000_file.replace('_top3000.txt', '')
            category_dir = os.path.join(quickdraw_jpg_dir, category)
            
            # Verificar si existe el directorio de la categoría
            if not os.path.exists(category_dir):
                print(f"Warning: Category directory not found: {category_dir}")
                continue
                
            # Leer IDs del archivo top3000
            with open(os.path.join(top3000_dir, top3000_file), 'r') as f:
                # Cada línea tiene el formato "ID.npy METRICA"
                image_ids = []
                for line in f:
                    # Tomar solo la primera parte antes del espacio y convertir .npy a .jpg
                    img_id = line.strip().split()[0].replace('.npy', '.jpg')
                    image_ids.append(img_id)
            
            # Agregar rutas completas de las imágenes
            category_images = 0
            for img_id in image_ids:
                img_path = os.path.join(category_dir, img_id)
                if os.path.exists(img_path):
                    self.image_paths.append(img_path)
                    category_images += 1
            
            print(f"Category {category}: loaded {category_images} images")
        
        if len(self.image_paths) == 0:
            raise ValueError("No images found! Please check if the image paths are correct.")
            
        print(f"Total: Loaded {len(self.image_paths)} images from {len(self.top3000_files)/2} categories")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Cargar y preprocesar imagen
        image = Image.open(img_path).convert('L')  # Convertir a escala de grises
        
        # Redimensionar
        image = image.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        
        # Convertir a tensor y normalizar a [-1, 1]
        image = torch.from_numpy(np.array(image)).float()
        image = image.unsqueeze(0)  # Agregar dimensión de canal (H,W) -> (1,H,W)
        image = (image / 127.5) - 1.0  # Normalizar a [-1, 1]
        
        return {"target": image} 