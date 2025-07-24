#!/usr/bin/env python3
"""
Script para usar el modelo DDPM entrenado para generar imágenes de sketches.
"""

import os
import torch
import argparse
import numpy as np
from diffusers import DDPMScheduler
from utils.sampler import SamplingPipeline
from config import config
from PIL import Image

def load_starting_image(image_path, target_size=(128, 128)):
    """
    Carga y preprocesa la imagen de inicio.
    
    Args:
        image_path (str): Ruta a la imagen de inicio
        target_size (tuple): Tamaño objetivo de la imagen
    
    Returns:
        torch.Tensor: Imagen preprocesada como tensor
    """
    # Cargar imagen
    image = Image.open(image_path).convert('L')  # Convertir a escala de grises
    
    # Redimensionar
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convertir a numpy array y normalizar a [0, 1]
    image_array = np.array(image).astype(np.float32) / 255.0
    
    # Convertir a tensor y agregar dimensiones de batch y channel
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    return image_tensor

def generate_sketches(model_path, num_images=4, output_dir="generated_sketches", starting_image=None):
    """
    Genera imágenes de sketches usando el modelo DDPM entrenado.
    
    Args:
        model_path (str): Ruta al directorio del modelo entrenado
        num_images (int): Número de imágenes a generar
        output_dir (str): Directorio donde guardar las imágenes generadas
        starting_image (str): Ruta a la imagen de inicio (opcional)
    """
    
    # Verificar que el modelo existe
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"El directorio del modelo no existe: {model_path}")
    
    # Configurar dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")
    
    # Cargar el pipeline del modelo entrenado
    print(f"Cargando modelo desde: {model_path}")
    pipeline = SamplingPipeline.from_pretrained(
        model_path, 
        use_safetensors=True, 
        conditioning=config.conditioning
    ).to(device)
    
    # Configurar el scheduler
    pipeline.scheduler = config.scheduler.from_pretrained(f"{model_path}/scheduler/")
    
    # Preparar imagen de inicio si se especifica
    starting_tensor = None
    if starting_image and os.path.exists(starting_image):
        print(f"Cargando imagen de inicio: {starting_image}")
        starting_tensor = load_starting_image(starting_image)
        
        # Repetir la imagen para el batch size - asegurar que sea (batch, channel, height, width)
        starting_tensor = starting_tensor.expand(num_images, -1, -1, -1)
        
        # Convertir al formato que espera el preprocess method (diccionario con 'image' key)
        starting_tensor = {'image': starting_tensor}
        
        print(f"Imagen de inicio cargada y preparada para {num_images} imágenes")
    else:
        print("Generando desde ruido aleatorio")
    
    print(f"Generando {num_images} imágenes de sketches...")
    
    # Generar imágenes
    with torch.no_grad():
        output = pipeline(
            batch_size=num_images,
            images=starting_tensor,
            num_inference_steps=config.num_inference_steps,
            output_type='np.array'
        )
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar las imágenes generadas
    images = output.images
    for i, image in enumerate(images):
        # Asegurar que la imagen tenga la forma correcta
        if image.ndim == 3:
            if image.shape[0] == 1:  # Si es (1, H, W)
                image = image.squeeze(0)  # Convertir a (H, W)
            elif image.shape[2] == 1:  # Si es (H, W, 1)
                image = image.squeeze(2)  # Convertir a (H, W)
        
        # Normalizar a 0-255
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype('uint8')
        
        # Obtener el nombre base de la imagen original (sin extensión)
        if starting_image:
            base_name = os.path.splitext(os.path.basename(starting_image))[0]
            if num_images > 1:
                filename = f"sketch_{base_name}_{i+1:02d}.png"
            else:
                filename = f"sketch_{base_name}.png"
        else:
            filename = f"sketch_{i:04d}.png"
        
        # Guardar como imagen usando PIL para mejor compatibilidad
        pil_image = Image.fromarray(image, mode='L')  # 'L' para escala de grises
        pil_image.save(f"{output_dir}/{filename}")
    
    print(f"Se han generado {num_images} imágenes en el directorio: {output_dir}")
    if starting_image:
        if num_images > 1:
            print(f"Imágenes guardadas como: {output_dir}/sketch_[nombre_original]_XX.png")
        else:
            print(f"Imágenes guardadas como: {output_dir}/sketch_[nombre_original].png")
    else:
        print(f"Imágenes guardadas como: {output_dir}/sketch_XXXX.png")

def main():
    parser = argparse.ArgumentParser(description="Generar imágenes de sketches usando un modelo DDPM entrenado")
    parser.add_argument("--model_path", type=str, 
                       default="train/ddpm_sketches_test-no-128-1-42-2025-01-06-01:23",
                       help="Ruta al directorio del modelo entrenado")
    parser.add_argument("--num_images", type=int, default=4, 
                       help="Número de imágenes a generar")
    parser.add_argument("--output_dir", type=str, default="generated_sketches", 
                       help="Directorio donde guardar las imágenes generadas")
    parser.add_argument("--starting_image", type=str, default="29030_inverted.png",
                       help="Ruta a la imagen de inicio (opcional)")
    
    args = parser.parse_args()
    
    try:
        generate_sketches(args.model_path, args.num_images, args.output_dir, args.starting_image)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 