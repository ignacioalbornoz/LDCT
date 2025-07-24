#!/usr/bin/env python3
"""
Script para generar imágenes capturando los pasos intermedios reales del proceso de denoising.
"""

import os
import torch
import numpy as np
import argparse
from PIL import Image
from diffusers import DDPMScheduler
from utils.sampler import SamplingPipeline
from config import config

class StepCapturePipeline(SamplingPipeline):
    """Pipeline modificado para capturar pasos intermedios."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured_steps = {}
    
    def capture_step(self, step_idx, latents):
        """Captura el estado de un paso específico."""
        # Convertir latents a imagen
        with torch.no_grad():
            # Para DDPM, los latents son directamente las imágenes
            image = latents.cpu().numpy()
            if image.ndim == 4:
                image = image[0]  # Tomar el primer batch
            self.captured_steps[step_idx] = image.copy()
    
    @torch.no_grad()
    def generate_with_step_capture(self, batch_size=1, images=None, num_inference_steps=1000, 
                                 target_steps=None, output_type='np.array'):
        """Genera imágenes capturando pasos específicos."""
        
        # Configurar scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        
        # Preparar imagen inicial
        if images is None:
            # Generar ruido aleatorio
            if isinstance(self.unet.config.sample_size, int):
                image_shape = (
                    batch_size,
                    self.unet.config.in_channels,
                    self.unet.config.sample_size,
                    self.unet.config.sample_size,
                )
            else:
                in_channels = self.unet.config.in_channels//2 if self.conditioning is not None else self.unet.config.in_channels
                image_shape = (batch_size, in_channels, *self.unet.config.sample_size)
            
            latents = torch.randn(image_shape, device=self.device, dtype=self.unet.dtype)
        else:
            # Usar imagen de entrada
            images = self.preprocess(images)
            images = images.to(self.device)
            
            # Agregar ruido según el proceso de difusión
            if isinstance(self.inverse_scheduler, str) and self.inverse_scheduler == "default":
                noise = torch.randn(images.shape).to(self.device)
                latents = self.scheduler.add_noise(images, noise, timesteps[0])
            else:
                latents = images
        
        # Capturar paso inicial (paso 0)
        if target_steps and 0 in target_steps:
            self.capture_step(0, latents)
        
        # Proceso de denoising
        for i, t in enumerate(timesteps):
            # Predecir ruido residual
            latent_model_input = self.scheduler.scale_model_input(latents, t)
            noise_pred = self.unet(latent_model_input, t).sample
            
            # Calcular el paso anterior
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            # Capturar pasos específicos
            current_step = num_inference_steps - i - 1
            if target_steps and current_step in target_steps:
                self.capture_step(current_step, latents)
        
        # Capturar paso final
        if target_steps and num_inference_steps in target_steps:
            self.capture_step(num_inference_steps, latents)
        
        return self.captured_steps

def generate_with_intermediate_steps(model_path, starting_image=None, output_dir="intermediate_steps", 
                                   target_steps=[0, 1, 2, 3, 4, 5, 19, 40, 65, 95, 129, 168, 215, 269, 331, 405, 490, 590, 706, 842, 1000]):
    """
    Genera imágenes capturando los pasos intermedios reales del proceso de denoising.
    """
    
    # Verificar que el modelo existe
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"El directorio del modelo no existe: {model_path}")
    
    # Configurar dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")
    
    # Cargar el pipeline del modelo entrenado
    print(f"Cargando modelo desde: {model_path}")
    pipeline = StepCapturePipeline.from_pretrained(
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
    
    # Generar imágenes capturando pasos intermedios
    captured_steps = pipeline.generate_with_step_capture(
        batch_size=1,
        images=starting_tensor,
        num_inference_steps=1000,
        target_steps=target_steps
    )
    
    # Guardar imágenes capturadas
    for step, image in captured_steps.items():
        try:
            # Normalizar la imagen
            if image.ndim == 4:
                image = image[0]  # Tomar el primer batch
            
            if image.ndim == 3:
                if image.shape[0] == 1:
                    image = image.squeeze(0)
                elif image.shape[2] == 1:
                    image = image.squeeze(2)
            
            # Normalizar a 0-255
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype('uint8')
            
            # Guardar imagen
            filename = f"step_{step:04d}.png"
            pil_image = Image.fromarray(image, mode='L')
            pil_image.save(f"{output_dir}/{filename}")
            
            print(f"✓ Paso {step} guardado como {filename}")
            
        except Exception as e:
            print(f"✗ Error guardando paso {step}: {e}")
            continue
    
    print(f"\n¡Proceso completado!")
    print(f"Imágenes generadas guardadas en: {output_dir}")

def load_starting_image(image_path, target_size=(128, 128)):
    """Carga y preprocesa la imagen de inicio."""
    image = Image.open(image_path).convert('L')
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    image_array = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)
    return image_tensor

def main():
    parser = argparse.ArgumentParser(description="Generar imágenes capturando pasos intermedios del proceso de difusión")
    parser.add_argument("--model_path", type=str, 
                       default="train/ddpm_sketches_test-no-128-1-42-2025-01-06-01:23",
                       help="Ruta al directorio del modelo entrenado")
    parser.add_argument("--starting_image", type=str, default=None,
                       help="Ruta a la imagen de inicio (opcional)")
    parser.add_argument("--output_dir", type=str, default="intermediate_steps",
                       help="Directorio donde guardar las imágenes generadas")
    parser.add_argument("--steps", type=str, default="0,1,2,3,4,5,19,40,65,95,129,168,215,269,331,405,490,590,706,842,1000",
                       help="Pasos específicos a mostrar (separados por comas)")
    
    args = parser.parse_args()
    
    # Convertir string de pasos a lista de enteros
    target_steps = [int(s.strip()) for s in args.steps.split(',')]
    
    try:
        generate_with_intermediate_steps(
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