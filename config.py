import time

from dataclasses import dataclass

from diffusers import DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler, DPMSolverSDEScheduler, UniPCMultistepScheduler#, EDMEulerScheduler
from diffusers import DDPMPipeline, DDIMPipeline

@dataclass
class TrainingConfig:
	seed = 42
	
	image_size = 128  # Reduced from 256 to save memory
	
	train_batch_size = 8  # Reduced from 16 to save memory
	eval_batch_size = 4
	
	num_epochs = 10  # Reduced for quick testing
	num_train_timesteps = 200  # Reduced from 1000 for faster training
	num_inference_steps = 200  # Match with train timesteps
	
	model_name = "DDPM_Sketches_Test"
	scheduler = DDPMScheduler
	pipeline = DDPMPipeline
	
	conditioning = None  # No conditioning for sketch generation
	
	slices = 1
	channels = 1  # Grayscale images
	
	learning_rate = 1e-4
	lr_warmup_steps = 100  # Reduced to match shorter training
	
	save_image_epochs = 1  # Save images every epoch to monitor progress
	save_model_epochs = 1
	
	mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
	
	gradient_accumulation_steps = 2  # Increased to compensate for smaller batch size
	
	push_to_hub = False
	hub_private_repo = False
	overwrite_output_dir = False
	
	output_dir = f"train/{model_name.lower()}-{mixed_precision}-{image_size}-{slices}-{seed}-{time.strftime('%Y-%d-%m-%H:%M', time.localtime(time.time()))}"

config = TrainingConfig()
