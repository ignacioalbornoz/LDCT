#!/usr/bin/env python3
"""
Script para generar imágenes de sketches usando el modelo entrenado
para cada imagen en la carpeta edges_contours.
"""

import os
import glob
from use_trained_model import generate_sketches

def generate_from_edges_folder(model_path, edges_folder="edges_contours", output_folder="generated_from_edges", num_images_per_input=5):
    """
    Genera imágenes de sketches para cada imagen en la carpeta edges_contours.
    
    Args:
        model_path (str): Ruta al directorio del modelo entrenado
        edges_folder (str): Carpeta con las imágenes de entrada
        output_folder (str): Carpeta donde guardar las imágenes generadas
        num_images_per_input (int): Número de imágenes a generar por cada imagen de entrada
    """
    
    # Verificar que la carpeta de entrada existe
    if not os.path.exists(edges_folder):
        raise FileNotFoundError(f"La carpeta {edges_folder} no existe")
    
    # Crear carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)
    
    # Obtener todas las imágenes PNG en la carpeta
    image_files = glob.glob(os.path.join(edges_folder, "*.png"))
    
    if not image_files:
        print(f"No se encontraron imágenes PNG en {edges_folder}")
        return
    
    print(f"Encontradas {len(image_files)} imágenes en {edges_folder}")
    print(f"Generando {num_images_per_input} imágenes por cada imagen de entrada...")
    
    # Procesar cada imagen
    for i, image_path in enumerate(image_files):
        print(f"\nProcesando imagen {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        try:
            # Generar imágenes usando el modelo entrenado
            generate_sketches(
                model_path=model_path,
                num_images=num_images_per_input,
                output_dir=output_folder,
                starting_image=image_path
            )
            print(f"✓ Completado: {os.path.basename(image_path)}")
            
        except Exception as e:
            print(f"✗ Error procesando {os.path.basename(image_path)}: {e}")
            continue
    
    print(f"\n¡Proceso completado!")
    print(f"Imágenes generadas guardadas en: {output_folder}")
    print(f"Total de imágenes procesadas: {len(image_files)}")
    print(f"Total de imágenes generadas: {len(image_files) * num_images_per_input}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generar imágenes de sketches para cada imagen en edges_contours")
    parser.add_argument("--model_path", type=str, 
                       default="train/ddpm_sketches_test-no-128-1-42-2025-01-06-01:23",
                       help="Ruta al directorio del modelo entrenado")
    parser.add_argument("--edges_folder", type=str, default="edges_contours",
                       help="Carpeta con las imágenes de entrada")
    parser.add_argument("--output_folder", type=str, default="generated_from_edges",
                       help="Carpeta donde guardar las imágenes generadas")
    parser.add_argument("--num_images_per_input", type=int, default=5,
                       help="Número de imágenes a generar por cada imagen de entrada")
    
    args = parser.parse_args()
    
    try:
        generate_from_edges_folder(
            model_path=args.model_path,
            edges_folder=args.edges_folder,
            output_folder=args.output_folder,
            num_images_per_input=args.num_images_per_input
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 