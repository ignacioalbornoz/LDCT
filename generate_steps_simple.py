#!/usr/bin/env python3
"""
Script simple para generar imágenes mostrando pasos específicos del proceso de difusión.
"""

import os
import torch
import numpy as np
import argparse
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler
from utils.sampler import SamplingPipeline
from config import config

def generate_steps_simple(model_path, starting_image=None, output_dir="steps_simple", 
                         target_steps=[0, 1, 2, 3, 4, 5, 19, 40, 65, 95, 129, 168, 215, 269, 331, 405, 490, 590, 706, 842, 1000]):
    """
    Genera imágenes para pasos específicos usando diferentes números de pasos de inferencia.
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
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Preparar imagen de inicio si se especifica
    starting_tensor = None
    if starting_image and os.path.exists(starting_image):
        print(f"Cargando imagen de inicio: {starting_image}")
        starting_tensor = load_starting_image(starting_image)
        starting_tensor = {'image': starting_tensor}
        print(f"Imagen de inicio cargada")
    else:
        print("Generando desde ruido aleatorio")
    
    print(f"Generando imágenes para {len(target_steps)} pasos específicos...")
    
    # Lista para almacenar todas las imágenes generadas
    all_images = []
    all_steps = []
    
    # Generar imágenes para cada paso específico
    for step in target_steps:
        print(f"Generando paso {step}...")
        
        try:
            if step == 0:
                # Paso 0: imagen original (sin ruido)
                if starting_tensor is not None:
                    image = starting_tensor['image'].cpu().numpy().squeeze()
                else:
                    # Si no hay imagen de inicio, generar con muy pocos pasos
                    image = generate_with_steps(pipeline, starting_tensor, 1, device)
            else:
                # Otros pasos: generar con el número de pasos correspondiente
                image = generate_with_steps(pipeline, starting_tensor, step, device)
            
            # Normalizar la imagen
            if image.ndim == 3:
                if image.shape[0] == 1:
                    image = image.squeeze(0)
                elif image.shape[2] == 1:
                    image = image.squeeze(2)
            
            # Normalizar a 0-255
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype('uint8')
            
            # Guardar imagen individual
            filename = f"step_{step:04d}.png"
            pil_image = Image.fromarray(image, mode='L')
            pil_image.save(f"{output_dir}/{filename}")
            
            # Almacenar para la imagen combinada
            all_images.append(image)
            all_steps.append(step)
            
            print(f"✓ Paso {step} guardado como {filename}")
            
        except Exception as e:
            print(f"✗ Error en paso {step}: {e}")
            continue
    
    # Crear imagen combinada con todos los pasos
    if all_images:
        print("Creando imagen combinada con todos los pasos...")
        create_combined_image(all_images, all_steps, output_dir)
    
    print(f"\n¡Proceso completado!")
    print(f"Imágenes individuales guardadas en: {output_dir}")
    print(f"Imagen combinada guardada como: {output_dir}/all_steps_combined.png")

def create_combined_image(images, steps, output_dir):
    """Crea una imagen que muestra todos los pasos juntos."""
    
    # Calcular el número de filas y columnas para la cuadrícula
    n_images = len(images)
    n_cols = 7  # 7 columnas por fila
    n_rows = (n_images + n_cols - 1) // n_cols  # Calcular filas necesarias
    
    # Crear figura
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    
    # Asegurar que axes sea siempre un array 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Mostrar cada imagen
    for i, (image, step) in enumerate(zip(images, steps)):
        row = i // n_cols
        col = i % n_cols
        
        axes[row, col].imshow(image, cmap='gray')
        axes[row, col].set_title(f"Step {step}")
        axes[row, col].axis('off')
    
    # Ocultar subplots vacíos
    for i in range(n_images, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/all_steps_combined.png", dpi=150, bbox_inches='tight')
    plt.close()

def load_starting_image(image_path, target_size=(128, 128)):
    """Carga y preprocesa la imagen de inicio."""
    image = Image.open(image_path).convert('L')
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    image_array = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)
    return image_tensor

def generate_with_steps(pipeline, starting_tensor, num_steps, device):
    """Genera una imagen con un número específico de pasos."""
    # Configurar el scheduler para el número de pasos objetivo
    pipeline.scheduler.set_timesteps(num_steps)
    
    # Generar imagen con el número de pasos específico
    with torch.no_grad():
        output = pipeline(
            batch_size=1,
            images=starting_tensor,
            num_inference_steps=num_steps,
            output_type='np.array'
        )
    
    return output.images[0]

def main():
    parser = argparse.ArgumentParser(description="Generar imágenes para pasos específicos del proceso de difusión")
    parser.add_argument("--model_path", type=str, 
                       default="train/ddpm_sketches_test-no-128-1-42-2025-01-06-01:23",
                       help="Ruta al directorio del modelo entrenado")
    parser.add_argument("--starting_image", type=str, default=None,
                       help="Ruta a la imagen de inicio (opcional)")
    parser.add_argument("--output_dir", type=str, default="steps_simple",
                       help="Directorio donde guardar las imágenes generadas")
    parser.add_argument("--steps", type=str, default="0,1,2,3,4,5,19,40,65,95,129,168,215,269,331,405,490,590,706,842,1000",
                       help="Pasos específicos a mostrar (separados por comas)")
    
    args = parser.parse_args()
    
    # Convertir string de pasos a lista de enteros
    target_steps = [int(s.strip()) for s in args.steps.split(',')]
    
    try:
        generate_steps_simple(
            model_path=args.model_path,
            starting_image=args.starting_image,
            output_dir=args.output_dir,
            target_steps=target_steps
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 