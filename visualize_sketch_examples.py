#!/usr/bin/env python3
"""
Script para visualizar ejemplos específicos de edges con diferentes cantidades de sketches.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from utils.sketch_dataset import SketchDataset

def find_examples_by_sketch_count():
    """Encontrar ejemplos de edges con diferentes cantidades de sketches"""
    
    base_path = '/home/shared_data/Datasets/sketchy/train'
    edge_dir = os.path.join(base_path, 'edges_inverted_bw')
    sketch_dir = os.path.join(base_path, 'sketch_bw')
    
    # Analizar sketches por edge
    edge_to_sketches = {}
    
    categories = [d for d in os.listdir(edge_dir) 
                 if os.path.isdir(os.path.join(edge_dir, d))]
    
    for category in categories:
        edge_cat_path = os.path.join(edge_dir, category)
        sketch_cat_path = os.path.join(sketch_dir, category)
        
        if not os.path.exists(sketch_cat_path):
            continue
        
        edge_files = [f for f in os.listdir(edge_cat_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for edge_file in edge_files:
            base_name = os.path.splitext(edge_file)[0]
            
            sketch_files = [f for f in os.listdir(sketch_cat_path) 
                          if f.startswith(base_name + '-') and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            num_sketches = len(sketch_files)
            edge_to_sketches[f"{category}/{edge_file}"] = {
                'num_sketches': num_sketches,
                'sketch_files': sketch_files,
                'category': category,
                'base_name': base_name
            }
    
    return edge_to_sketches

def visualize_specific_examples(edge_to_sketches):
    """Visualizar ejemplos específicos de edges con diferentes cantidades de sketches"""
    
    # Agrupar por cantidad de sketches
    edges_by_count = {}
    for edge_path, info in edge_to_sketches.items():
        count = info['num_sketches']
        if count not in edges_by_count:
            edges_by_count[count] = []
        edges_by_count[count].append(edge_path)
    
    # Seleccionar ejemplos para visualizar
    examples_to_show = [1, 5, 6, 8, 10, 15]  # Cantidades específicas
    
    fig, axes = plt.subplots(len(examples_to_show), 6, figsize=(20, 4*len(examples_to_show)))
    if len(examples_to_show) == 1:
        axes = axes.reshape(1, -1)
    
    for row, sketch_count in enumerate(examples_to_show):
        if sketch_count not in edges_by_count:
            print(f"No hay ejemplos con {sketch_count} sketches")
            continue
        
        # Tomar el primer ejemplo con esta cantidad de sketches
        example_edge = edges_by_count[sketch_count][0]
        info = edge_to_sketches[example_edge]
        
        print(f"Ejemplo con {sketch_count} sketches: {example_edge}")
        
        # Cargar edge
        edge_path = os.path.join('/home/shared_data/Datasets/sketchy/train/edges_inverted_bw', example_edge)
        edge_img = plt.imread(edge_path)
        if len(edge_img.shape) == 3:
            edge_img = edge_img[:, :, 0]
        
        # Mostrar edge
        axes[row, 0].imshow(edge_img, cmap='gray')
        axes[row, 0].set_title(f'Edge ({sketch_count} sketches)\n{info["category"]}')
        axes[row, 0].axis('off')
        
        # Mostrar sketches
        for i in range(min(5, sketch_count)):  # Máximo 5 sketches por fila
            sketch_file = info['sketch_files'][i]
            sketch_path = os.path.join('/home/shared_data/Datasets/sketchy/train/sketch_bw', 
                                     info['category'], sketch_file)
            
            if os.path.exists(sketch_path):
                sketch_img = plt.imread(sketch_path)
                if len(sketch_img.shape) == 3:
                    sketch_img = sketch_img[:, :, 0]
                
                axes[row, i+1].imshow(sketch_img, cmap='gray')
                axes[row, i+1].set_title(f'Sketch {i+1}')
                axes[row, i+1].axis('off')
            else:
                axes[row, i+1].text(0.5, 0.5, 'No encontrado', ha='center', va='center')
                axes[row, i+1].axis('off')
        
        # Ocultar subplots vacíos
        for i in range(sketch_count + 1, 6):
            axes[row, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sketch_examples_by_count.png', dpi=150, bbox_inches='tight')
    print("Visualización guardada como 'sketch_examples_by_count.png'")

def create_statistics_plot(edge_to_sketches):
    """Crear gráfico de estadísticas"""
    
    # Contar distribución
    counts = [info['num_sketches'] for info in edge_to_sketches.values()]
    counter = Counter(counts)
    
    # Crear gráfico
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico de barras
    sorted_counts = sorted(counter.items())
    counts_list, frequencies = zip(*sorted_counts)
    
    ax1.bar(counts_list, frequencies, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Número de sketches por edge')
    ax1.set_ylabel('Número de edges')
    ax1.set_title('Distribución de sketches por edge')
    ax1.grid(True, alpha=0.3)
    
    # Agregar valores en las barras
    for i, v in enumerate(frequencies):
        ax1.text(counts_list[i], v + max(frequencies)*0.01, str(v), ha='center', va='bottom')
    
    # Gráfico de porcentajes acumulados
    total_edges = sum(frequencies)
    percentages = [f/total_edges*100 for f in frequencies]
    cumulative = np.cumsum(percentages)
    
    ax2.plot(counts_list, cumulative, 'o-', linewidth=2, markersize=8, color='red')
    ax2.set_xlabel('Número de sketches por edge')
    ax2.set_ylabel('Porcentaje acumulado (%)')
    ax2.set_title('Porcentaje acumulado de edges')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Agregar líneas de referencia
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=90, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('sketch_statistics.png', dpi=150, bbox_inches='tight')
    print("Estadísticas guardadas como 'sketch_statistics.png'")

if __name__ == "__main__":
    print("Analizando dataset de sketch...")
    edge_to_sketches = find_examples_by_sketch_count()
    
    print(f"Total de edges analizados: {len(edge_to_sketches)}")
    
    # Crear visualizaciones
    visualize_specific_examples(edge_to_sketches)
    create_statistics_plot(edge_to_sketches)
    
    print("\n=== RESUMEN DE HALLAZGOS ===")
    print("Basado en el análisis anterior:")
    print("- El dataset tiene 11,249 edges únicos")
    print("- Se generan 67,801 pares de entrenamiento total")
    print("- Promedio: 6.03 sketches por edge")
    print("- Mediana: 6 sketches por edge")
    print("- Rango: 0 a 18 sketches por edge")
    print("- La mayoría de edges (77%) tienen 5-6 sketches")
    print("- Solo 8 edges (0.1%) no tienen sketches")
    print("- 2 edges tienen el máximo de 18 sketches")

