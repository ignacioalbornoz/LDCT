import os
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate import notebook_launcher
from models.DiffUNet2D import model as Unet2D
from diffusers.utils import make_image_grid
from huggingface_hub import HfFolder, Repository, whoami
from diffusers.optimization import get_cosine_schedule_with_warmup
from utils.quickdraw_dataset import QuickDrawDataset
from utils.sampler import SamplingPipeline
from config import config

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    print("\n=== Starting Training Process ===")
    print(f"Training configuration:")
    print(f"- Number of epochs: {config.num_epochs}")
    print(f"- Batch size: {config.train_batch_size}")
    print(f"- Learning rate: {config.learning_rate}")
    print(f"- Mixed precision: {config.mixed_precision}")
    print("===============================\n")

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    
    if accelerator.is_main_process:
        if config.push_to_hub:
            print("Pushing to Hugging Face Hub...")
            repo_name = get_full_repo_name(Path(config.output_dir).name)
            repo = Repository(config.output_dir, clone_from=repo_name)
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
            print(f"Output directory created at: {config.output_dir}")
        accelerator.init_trackers("train_example")
        
    # Prepare everything
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    global_step = 0
    
    # Now you train the model
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Training")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["target"]
            
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]
            
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()
            
            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
            
        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            print(f"\nEvaluating epoch {epoch + 1}...")
            pipeline = SamplingPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                print("Generating sample images...")
                evaluate(config, epoch, pipeline)
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                print("Saving model checkpoint...")
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    pipeline.save_pretrained(config.output_dir)
                print(f"Model saved at: {config.output_dir}")

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise
    images = pipeline(
        batch_size=config.eval_batch_size,
    ).images[:config.eval_batch_size]

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=config.eval_batch_size//4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
    print(f"Sample images saved at: {test_dir}/{epoch:04d}.png")

def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

if __name__ == '__main__':
    print("\n=== Initializing Training Setup ===")
    
    # Verificar rutas de datos
    top3000_path = '/data/top3000_by_category'
    quickdraw_jpg_path = '/data/quickdraw_jpg'
    
    print("Verifying data paths...")
    if not os.path.exists(top3000_path):
        raise ValueError(f"Directory not found: {top3000_path}")
    if not os.path.exists(quickdraw_jpg_path):
        raise ValueError(f"Directory not found: {quickdraw_jpg_path}")
    
    # Listar categorías disponibles
    categories = [f.split('_top3000.txt')[0] for f in os.listdir(top3000_path) 
                 if f.endswith('_top3000.txt') and not f.endswith('_worst100.txt')]
    print(f"Found {len(categories)} categories: {', '.join(categories[:5])}...")
    
    # Configurar el modelo y el scheduler
    print("\nLoading model and scheduler...")
    model = Unet2D
    noise_scheduler = config.scheduler(num_train_timesteps=config.num_train_timesteps)
    
    # Crear el dataset
    print("\nCreating dataset...")
    dataset = QuickDrawDataset(
        top3000_dir=top3000_path,
        quickdraw_jpg_dir=quickdraw_jpg_path,
        img_size=config.image_size,
        train=True
    )
    print(f"Dataset size: {len(dataset)} samples")
    if len(dataset) == 0:
        raise ValueError("Dataset is empty! Please check the data paths and file structure.")
    
    # Crear el dataloader
    print("\nSetting up data loader...")
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Configurar el optimizador y el scheduler de learning rate
    print("\nConfiguring optimizer and learning rate scheduler...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )
    
    # Iniciar el entrenamiento
    print("\nStarting training process...")
    args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
    notebook_launcher(train_loop, args, num_processes=1) 