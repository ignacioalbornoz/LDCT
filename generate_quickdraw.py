import os
import torch
from diffusers import DDPMScheduler
from models.DiffUNet2D import model as Unet2D
from utils.sampler import SamplingPipeline
from config import config

def generate_images(model_path, num_images=16, output_dir="generated_images"):
    # Cargar el modelo entrenado
    model = Unet2D
    model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin")))
    model.eval()
    
    # Configurar el scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps)
    
    # Crear el pipeline de generación
    pipeline = SamplingPipeline(unet=model, scheduler=noise_scheduler)
    
    # Generar imágenes
    images = pipeline(
        batch_size=num_images,
    ).images
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar las imágenes generadas
    for i, image in enumerate(images):
        image.save(os.path.join(output_dir, f"generated_{i:04d}.png"))
    
    print(f"Se han generado {num_images} imágenes en el directorio {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generar imágenes de sketches usando un modelo de difusión entrenado")
    parser.add_argument("--model_path", type=str, required=True, help="Ruta al directorio del modelo entrenado")
    parser.add_argument("--num_images", type=int, default=16, help="Número de imágenes a generar")
    parser.add_argument("--output_dir", type=str, default="generated_images", help="Directorio donde guardar las imágenes generadas")
    
    args = parser.parse_args()
    
    generate_images(args.model_path, args.num_images, args.output_dir) 