#!/usr/bin/env python3
"""
Script simple para extraer contornos fuertes tomando solo los píxeles más oscuros.
Procesa todas las imágenes de una carpeta.
"""

import os
import numpy as np
from PIL import Image
import glob

def extract_darkest_pixels(input_path, output_path, percentile=10):
    """
    Extrae solo los píxeles más oscuros de la imagen.
    
    Args:
        input_path: Ruta de la imagen de entrada
        output_path: Ruta donde guardar la imagen procesada
        percentile: Porcentaje de píxeles más oscuros a mantener (0-100)
    """
    
    # Cargar la imagen
    print(f"Procesando: {os.path.basename(input_path)}")
    img = Image.open(input_path)
    
    # Convertir a escala de grises si no lo está
    if img.mode != 'L':
        img = img.convert('L')
    
    # Convertir a numpy array
    img_array = np.array(img)
    
    # Calcular el umbral basado en el percentil
    threshold = np.percentile(img_array, percentile)
    
    # Crear máscara: solo los píxeles más oscuros que el umbral
    mask = img_array <= threshold
    
    # Crear imagen resultante: negro para píxeles oscuros, blanco para el resto
    result = np.ones_like(img_array) * 255  # Fondo blanco
    result[mask] = 0  # Píxeles oscuros en negro
    
    # Convertir de vuelta a PIL Image
    result_img = Image.fromarray(result.astype(np.uint8))
    
    # Guardar resultado
    result_img.save(output_path)
    
    return result_img

def process_folder(input_folder, output_folder, percentile=10):
    """
    Procesa todas las imágenes de una carpeta.
    
    Args:
        input_folder: Carpeta con las imágenes de entrada
        output_folder: Carpeta donde guardar las imágenes procesadas
        percentile: Porcentaje de píxeles más oscuros a mantener
    """
    
    # Crear carpeta de salida si no existe
    os.makedirs(output_folder, exist_ok=True)
    
    # Obtener todas las imágenes PNG de la carpeta de entrada
    input_pattern = os.path.join(input_folder, "*.png")
    image_files = glob.glob(input_pattern)
    
    if not image_files:
        print(f"No se encontraron imágenes PNG en: {input_folder}")
        return
    
    print(f"Encontradas {len(image_files)} imágenes para procesar")
    print("=" * 50)
    
    # Procesar cada imagen
    for i, input_path in enumerate(image_files, 1):
        # Obtener nombre del archivo sin extensión
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_name = f"{base_name}_contours.png"
        output_path = os.path.join(output_folder, output_name)
        
        print(f"[{i}/{len(image_files)}] {base_name}")
        
        try:
            extract_darkest_pixels(input_path, output_path, percentile)
        except Exception as e:
            print(f"Error procesando {base_name}: {e}")
    
    print("=" * 50)
    print(f"Procesamiento completado. {len(image_files)} imágenes procesadas.")
    print(f"Resultados guardados en: {output_folder}")

if __name__ == "__main__":
    # Activar el entorno conda
    print("Asegúrate de tener activado el entorno: conda activate sketch_diffusion")
    
    # Configuración
    input_folder = "edges_inverted"
    output_folder = "edges_contours"
    percentile = 10  # Top 10% de píxeles más oscuros
    
    # Verificar que la carpeta de entrada existe
    if not os.path.exists(input_folder):
        print(f"Error: No se encontró la carpeta {input_folder}")
        exit(1)
    
    print("=" * 50)
    print("PROCESAMIENTO DE CARPETA DE IMÁGENES")
    print("=" * 50)
    print(f"Carpeta de entrada: {input_folder}")
    print(f"Carpeta de salida: {output_folder}")
    print(f"Percentil: {percentile}%")
    print("=" * 50)
    
    # Procesar todas las imágenes
    process_folder(input_folder, output_folder, percentile) 