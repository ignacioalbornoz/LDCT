import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class SketchPairDataset(Dataset):
    '''
    Dataset para pares (edge, sketch) en estructura tipo:
    root/edges_inverted_bw/<categoria>/<nombre>.png
    root/sketch/<categoria>/<nombre>-N.png
    '''
    def __init__(self, dataroot, img_size=256, norm=True, img_datatype=np.float32, transforms=None):
        super().__init__()
        self.dataroot = dataroot
        self.edges_dir = os.path.join(dataroot, 'edges_inverted_bw')
        self.sketch_dir = os.path.join(dataroot, 'sketch_bw')
        self.img_size = (img_size, img_size) if img_size is not None else None
        self.norm = norm
        self.img_datatype = img_datatype
        self.transforms = transforms
        
        # Crear pares de imágenes
        self.pairs = self._make_pairs()
        self.size = len(self.pairs)
        assert self.size > 0, 'Empty SketchPairDataset'
        
        print(f"[SketchPairDataset] Cargados {len(self.pairs)} pares de imágenes")
    
    def _make_pairs(self):
        pairs = []
        # Recorre todas las categorías
        for category in os.listdir(self.edges_dir):
            edge_cat_dir = os.path.join(self.edges_dir, category)
            sketch_cat_dir = os.path.join(self.sketch_dir, category)
            if not os.path.isdir(edge_cat_dir) or not os.path.isdir(sketch_cat_dir):
                continue
            # Indexa todos los sketches de la categoría
            sketch_files = os.listdir(sketch_cat_dir)
            for edge_file in os.listdir(edge_cat_dir):
                edge_prefix, _ = os.path.splitext(edge_file)
                # Busca todos los sketches que empiezan con el mismo prefijo
                matching_sketches = [f for f in sketch_files if f.startswith(edge_prefix + '-')]
                edge_path = os.path.join(edge_cat_dir, edge_file)
                for sketch_file in matching_sketches:
                    sketch_path = os.path.join(sketch_cat_dir, sketch_file)
                    pairs.append({
                        'edge_path': edge_path,
                        'sketch_path': sketch_path,
                        'category': category,
                        'edge_id': edge_prefix,
                        'sketch_file': sketch_file
                    })
        return pairs
    
    def __len__(self):
        return self.size

    def preprocess(self, img):
        # Convierte a float32, normaliza y resizea si es necesario
        img = np.array(img).astype(self.img_datatype)
        if self.img_size is not None:
            from skimage.transform import resize
            img = resize(img, self.img_size, preserve_range=True, anti_aliasing=True)
        
        # Convertir a binario (blanco/negro)
        if self.norm:
            # Umbral para convertir a binario
            threshold = 128
            img = (img > threshold).astype(np.float32)
        else:
            # Umbral para convertir a binario
            threshold = 128
            img = (img > threshold).astype(np.float32)
        return img

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Cargar imágenes en grayscale (1 canal)
        edge_img = Image.open(pair['edge_path']).convert('L')
        sketch_img = Image.open(pair['sketch_path']).convert('L')
        
        edge = self.preprocess(edge_img)
        sketch = self.preprocess(sketch_img)
        
        # Convertir a tensores [C, H, W] para grayscale
        edge = torch.as_tensor(edge).float().unsqueeze(0)  # [1, H, W]
        sketch = torch.as_tensor(sketch).float().unsqueeze(0)  # [1, H, W]
        
        if self.transforms is not None:
            edge, sketch = self.transforms(edge, sketch)
        
        return {
            'SR': edge,      # condición (edge) - 3 canales
            'HR': sketch     # objetivo (sketch) - 3 canales
        } 