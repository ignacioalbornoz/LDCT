#!/usr/bin/env python3
"""
Script para analizar el dataset de sketch y visualizar ejemplos.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from utils.sketch_dataset import SketchDataset

def analyze_sketch_dataset():
    """Analizar el dataset de sketch y mostrar estadísticas"""
    
    # Path al dataset
    base_path = '/home/shared_data/Datasets/sketchy/train'
    edge_dir = os.path.join(base_path, 'edges_inverted_bw')
    sketch_dir = os.path.join(base_path, 'sketch_bw')
    
    print(f"Analizando dataset en: {base_path}")
    print(f"Directorio de edges: {edge_dir}")
    print(f"Directorio de sketches: {sketch_dir}")
    
    # Verificar que los directorios existen
    if not os.path.exists(edge_dir):
        print(f"ERROR: No existe el directorio de edges: {edge_dir}")
        return
    
    if not os.path.exists(sketch_dir):
        print(f"ERROR: No existe el directorio de sketches: {sketch_dir}")
        return
    
    # Obtener categorías
    categories = [d for d in os.listdir(edge_dir) 
                 if os.path.isdir(os.path.join(edge_dir, d))]
    
    print(f"\nCategorías encontradas: {len(categories)}")
    print(f"Primeras 10 categorías: {categories[:10]}")
    
    # Analizar sketches por edge
    sketches_per_edge = []
    edge_to_sketches = {}
    
    for category in categories:
        edge_cat_path = os.path.join(edge_dir, category)
        sketch_cat_path = os.path.join(sketch_dir, category)
        
        if not os.path.exists(sketch_cat_path):
            continue
        
        # Obtener archivos de edge en esta categoría
        edge_files = [f for f in os.listdir(edge_cat_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for edge_file in edge_files:
            base_name = os.path.splitext(edge_file)[0]
            
            # Buscar sketches correspondientes
            sketch_files = [f for f in os.listdir(sketch_cat_path) 
                          if f.startswith(base_name + '-') and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            num_sketches = len(sketch_files)
            sketches_per_edge.append(num_sketches)
            
            edge_to_sketches[f"{category}/{edge_file}"] = {
                'num_sketches': num_sketches,
                'sketch_files': sketch_files
            }
    
    # Estadísticas
    print(f"\n=== ESTADÍSTICAS DEL DATASET ===")
    print(f"Total de edges analizados: {len(sketches_per_edge)}")
    print(f"Total de pares edge-sketch: {sum(sketches_per_edge)}")
    
    if sketches_per_edge:
        print(f"Mínimo sketches por edge: {min(sketches_per_edge)}")
        print(f"Máximo sketches por edge: {max(sketches_per_edge)}")
        print(f"Promedio sketches por edge: {np.mean(sketches_per_edge):.2f}")
        print(f"Mediana sketches por edge: {np.median(sketches_per_edge):.2f}")
        
        # Distribución
        counter = Counter(sketches_per_edge)
        print(f"\nDistribución de sketches por edge:")
        for num_sketches in sorted(counter.keys()):
            print(f"  {num_sketches} sketches: {counter[num_sketches]} edges ({counter[num_sketches]/len(sketches_per_edge)*100:.1f}%)")
    
    return edge_to_sketches, sketches_per_edge

def visualize_examples(edge_to_sketches, num_examples=5):
    """Visualizar ejemplos del dataset"""
    
    # Crear dataset para cargar imágenes
    dataset = SketchDataset(
        base_path='/home/shared_data/Datasets/sketchy/train',
        img_size=256,
        train=True,
        edge_dir='edges_inverted_bw',
        sketch_dir='sketch_bw'
    )
    
    print(f"\n=== VISUALIZANDO EJEMPLOS ===")
    print(f"Dataset total size: {len(dataset)}")
    
    # Encontrar edges con diferentes cantidades de sketches
    edges_with_1_sketch = [k for k, v in edge_to_sketches.items() if v['num_sketches'] == 1]
    edges_with_many_sketches = [k for k, v in edge_to_sketches.items() if v['num_sketches'] >= 5]
    
    print(f"Edges con 1 sketch: {len(edges_with_1_sketch)}")
    print(f"Edges con 5+ sketches: {len(edges_with_many_sketches)}")
    
    # Visualizar ejemplos
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Ejemplos del Dataset Sketch', fontsize=16)
    
    # Ejemplo 1: Edge con 1 sketch
    if edges_with_1_sketch:
        example_edge = edges_with_1_sketch[0]
        category, edge_file = example_edge.split('/', 1)
        base_name = os.path.splitext(edge_file)[0]
        
        # Encontrar el índice en el dataset
        for i in range(len(dataset)):
            sample = dataset[i]
            if base_name in sample['img_id']:
                edge_img = sample['image'].squeeze().numpy()
                sketch_img = sample['target'].squeeze().numpy()
                
                axes[0, 0].imshow(edge_img, cmap='gray')
                axes[0, 0].set_title(f'Edge (1 sketch)\n{category}')
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(sketch_img, cmap='gray')
                axes[0, 1].set_title('Sketch')
                axes[0, 1].axis('off')
                break
    
    # Ejemplo 2: Edge con muchos sketches
    if edges_with_many_sketches:
        example_edge = edges_with_many_sketches[0]
        category, edge_file = example_edge.split('/', 1)
        base_name = os.path.splitext(edge_file)[0]
        
        # Encontrar múltiples sketches para este edge
        sketch_files = edge_to_sketches[example_edge]['sketch_files'][:3]  # Primeros 3 sketches
        
        # Cargar edge
        edge_path = os.path.join('/home/shared_data/Datasets/sketchy/train/edges_inverted_bw', example_edge)
        edge_img = plt.imread(edge_path)
        if len(edge_img.shape) == 3:
            edge_img = edge_img[:, :, 0]  # Tomar solo el primer canal si es RGB
        
        axes[1, 0].imshow(edge_img, cmap='gray')
        axes[1, 0].set_title(f'Edge (múltiples sketches)\n{category}')
        axes[1, 0].axis('off')
        
        # Cargar sketches
        for i, sketch_file in enumerate(sketch_files):
            sketch_path = os.path.join('/home/shared_data/Datasets/sketchy/train/sketch_bw', example_edge.replace('.png', f'-{i+1}.png'))
            if os.path.exists(sketch_path):
                sketch_img = plt.imread(sketch_path)
                if len(sketch_img.shape) == 3:
                    sketch_img = sketch_img[:, :, 0]
                axes[1, i+1].imshow(sketch_img, cmap='gray')
                axes[1, i+1].set_title(f'Sketch {i+1}')
                axes[1, i+1].axis('off')
    
    # Ejemplo 3: Muestras aleatorias del dataset
    for i in range(4):
        idx = np.random.randint(0, len(dataset))
        sample = dataset[idx]
        
        edge_img = sample['image'].squeeze().numpy()
        sketch_img = sample['target'].squeeze().numpy()
        
        axes[2, i].imshow(edge_img, cmap='gray')
        axes[2, i].set_title(f'Muestra {i+1}\n{sample["img_id"][:30]}...')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sketch_dataset_analysis.png', dpi=150, bbox_inches='tight')
    print("Visualización guardada como 'sketch_dataset_analysis.png'")
    
    return dataset

if __name__ == "__main__":
    # Analizar dataset
    edge_to_sketches, sketches_per_edge = analyze_sketch_dataset()
    
    # Visualizar ejemplos
    dataset = visualize_examples(edge_to_sketches)
    
    print("\n=== ANÁLISIS COMPLETADO ===")
    print("Revisa el archivo 'sketch_dataset_analysis.png' para ver los ejemplos visuales.")

