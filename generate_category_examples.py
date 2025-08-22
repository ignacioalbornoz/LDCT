#!/usr/bin/env python3
"""
Script para generar 2 ejemplos aleatorios por cada categoría con todos sus sketches.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import defaultdict
from utils.sketch_dataset import SketchDataset

def get_category_examples():
    """Obtener ejemplos por categoría"""
    
    base_path = '/home/shared_data/Datasets/sketchy/train'
    edge_dir = os.path.join(base_path, 'edges_inverted_bw')
    sketch_dir = os.path.join(base_path, 'sketch_bw')
    
    # Organizar edges por categoría
    category_edges = defaultdict(list)
    
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
            
            if sketch_files:  # Solo incluir edges que tienen sketches
                category_edges[category].append({
                    'edge_file': edge_file,
                    'base_name': base_name,
                    'sketch_files': sketch_files,
                    'num_sketches': len(sketch_files)
                })
    
    return category_edges

def generate_category_visualizations(category_edges, output_dir='category_examples'):
    """Generar visualizaciones por categoría"""
    
    # Crear directorio de salida
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Generando ejemplos en: {output_dir}")
    
    # Configurar seed para reproducibilidad
    random.seed(42)
    
    base_path = '/home/shared_data/Datasets/sketchy/train'
    edge_dir = os.path.join(base_path, 'edges_inverted_bw')
    sketch_dir = os.path.join(base_path, 'sketch_bw')
    
    total_categories = len(category_edges)
    processed = 0
    
    for category, edges in category_edges.items():
        if len(edges) < 2:
            print(f"Saltando {category}: solo tiene {len(edges)} edges")
            continue
        
        print(f"Procesando {category} ({processed+1}/{total_categories})")
        
        # Seleccionar 2 ejemplos aleatorios
        selected_edges = random.sample(edges, 2)
        
        # Crear figura para esta categoría
        max_sketches = max(edge['num_sketches'] for edge in selected_edges)
        fig, axes = plt.subplots(2, max_sketches + 1, figsize=(3*(max_sketches + 1), 8))
        
        if max_sketches == 0:
            axes = axes.reshape(2, 1)
        
        for row, edge_info in enumerate(selected_edges):
            edge_file = edge_info['edge_file']
            sketch_files = edge_info['sketch_files']
            num_sketches = edge_info['num_sketches']
            
            # Cargar edge
            edge_path = os.path.join(edge_dir, category, edge_file)
            edge_img = plt.imread(edge_path)
            if len(edge_img.shape) == 3:
                edge_img = edge_img[:, :, 0]
            
            # Mostrar edge
            axes[row, 0].imshow(edge_img, cmap='gray')
            axes[row, 0].set_title(f'Edge ({num_sketches} sketches)')
            axes[row, 0].axis('off')
            
            # Mostrar sketches
            for i, sketch_file in enumerate(sketch_files):
                sketch_path = os.path.join(sketch_dir, category, sketch_file)
                
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
            for i in range(num_sketches + 1, max_sketches + 1):
                axes[row, i].axis('off')
        
        # Ajustar layout y guardar
        plt.suptitle(f'Categoría: {category}', fontsize=16)
        plt.tight_layout()
        
        # Guardar imagen
        output_path = os.path.join(output_dir, f'{category}_examples.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        processed += 1
    
    print(f"\nCompletado! Se generaron ejemplos para {processed} categorías.")
    print(f"Archivos guardados en: {output_dir}")

def create_summary_report(category_edges, output_dir='category_examples'):
    """Crear un reporte resumen de todas las categorías"""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Estadísticas por categoría
    category_stats = []
    for category, edges in category_edges.items():
        if edges:
            total_sketches = sum(edge['num_sketches'] for edge in edges)
            avg_sketches = total_sketches / len(edges)
            category_stats.append({
                'category': category,
                'num_edges': len(edges),
                'total_sketches': total_sketches,
                'avg_sketches': avg_sketches,
                'min_sketches': min(edge['num_sketches'] for edge in edges),
                'max_sketches': max(edge['num_sketches'] for edge in edges)
            })
    
    # Ordenar por número de edges
    category_stats.sort(key=lambda x: x['num_edges'], reverse=True)
    
    # Crear reporte
    report_path = os.path.join(output_dir, 'category_summary.txt')
    with open(report_path, 'w') as f:
        f.write("REPORTE DE CATEGORÍAS DEL DATASET SKETCH\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total de categorías: {len(category_stats)}\n")
        f.write(f"Total de edges: {sum(stat['num_edges'] for stat in category_stats)}\n")
        f.write(f"Total de sketches: {sum(stat['total_sketches'] for stat in category_stats)}\n\n")
        
        f.write("TOP 10 CATEGORÍAS CON MÁS EDGES:\n")
        f.write("-" * 40 + "\n")
        for i, stat in enumerate(category_stats[:10]):
            f.write(f"{i+1:2d}. {stat['category']:<15} | {stat['num_edges']:4d} edges | "
                   f"{stat['avg_sketches']:5.1f} avg sketches | "
                   f"{stat['min_sketches']:2d}-{stat['max_sketches']:2d} range\n")
        
        f.write("\nTODAS LAS CATEGORÍAS:\n")
        f.write("-" * 40 + "\n")
        for stat in category_stats:
            f.write(f"{stat['category']:<20} | {stat['num_edges']:4d} edges | "
                   f"{stat['avg_sketches']:5.1f} avg sketches | "
                   f"{stat['min_sketches']:2d}-{stat['max_sketches']:2d} range\n")
    
    print(f"Reporte guardado en: {report_path}")
    
    # Crear gráfico de distribución
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico de edges por categoría (top 20)
    top_20 = category_stats[:20]
    categories = [stat['category'] for stat in top_20]
    edge_counts = [stat['num_edges'] for stat in top_20]
    
    ax1.barh(range(len(categories)), edge_counts, color='skyblue')
    ax1.set_yticks(range(len(categories)))
    ax1.set_yticklabels(categories)
    ax1.set_xlabel('Número de edges')
    ax1.set_title('Top 20 Categorías por Número de Edges')
    ax1.invert_yaxis()
    
    # Gráfico de promedio de sketches por categoría
    avg_sketches = [stat['avg_sketches'] for stat in category_stats]
    ax2.hist(avg_sketches, bins=20, alpha=0.7, color='lightcoral', edgecolor='darkred')
    ax2.set_xlabel('Promedio de sketches por edge')
    ax2.set_ylabel('Número de categorías')
    ax2.set_title('Distribución del Promedio de Sketches por Categoría')
    ax2.axvline(np.mean(avg_sketches), color='red', linestyle='--', 
                label=f'Promedio: {np.mean(avg_sketches):.2f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico de distribución guardado en: {output_dir}/category_distribution.png")

if __name__ == "__main__":
    print("Generando ejemplos por categoría...")
    
    # Obtener ejemplos por categoría
    category_edges = get_category_examples()
    
    print(f"Encontradas {len(category_edges)} categorías")
    
    # Generar visualizaciones
    generate_category_visualizations(category_edges)
    
    # Crear reporte
    create_summary_report(category_edges)
    
    print("\n=== PROCESO COMPLETADO ===")
    print("Revisa la carpeta 'category_examples' para ver todos los ejemplos.")

