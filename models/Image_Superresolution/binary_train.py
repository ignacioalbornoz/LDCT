import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import time
from collections import OrderedDict
from copy import deepcopy
import matplotlib.pyplot as plt
import wandb

from data import create_dataset, create_dataloader
from model.model import create_model
from utils.logger import Logger, get_root_logger
from core.metrics import calculate_psnr, calculate_ssim
import utils.util as util
from utils.util import opt_get
import utils.option as option

class ConvergenceMonitor:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.losses = []
        self.gradients = []
        self.learning_rates = []
        self.psnr_values = []
        self.ssim_values = []
        self.image_quality = []
        
        # Early stopping
        self.best_psnr = 0
        self.best_step = 0
        self.patience = 50  # steps without improvement
        self.patience_counter = 0
        
        # Crear directorio para logs
        os.makedirs(log_dir, exist_ok=True)
        
    def log_loss(self, loss, step):
        self.losses.append((step, loss))
        
    def log_gradients(self, model, step):
        total_norm = 0
        param_count = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += p.numel()
        total_norm = total_norm ** (1. / 2)
        self.gradients.append((step, total_norm))
        
    def log_lr(self, lr, step):
        self.learning_rates.append((step, lr))
        
    def log_metrics(self, psnr, ssim, step):
        self.psnr_values.append((step, psnr))
        self.ssim_values.append((step, ssim))
        
        # Early stopping check
        if psnr > self.best_psnr:
            self.best_psnr = psnr
            self.best_step = step
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
    def should_stop(self):
        return self.patience_counter >= self.patience
        
    def log_image_quality(self, sr_img, hr_img, step):
        # Calcular métricas de calidad de imagen
        binary_accuracy = np.mean((sr_img > 0.5) == (hr_img > 0.5))
        edge_consistency = self.calculate_edge_consistency(sr_img, hr_img)
        self.image_quality.append((step, binary_accuracy, edge_consistency))
        
    def calculate_edge_consistency(self, sr_img, hr_img):
        # Calcular consistencia de edges usando detección de bordes
        from scipy import ndimage
        sr_edges = ndimage.sobel(sr_img)
        hr_edges = ndimage.sobel(hr_img)
        return np.corrcoef(sr_edges.flatten(), hr_edges.flatten())[0, 1]
        
    def save_plots(self):
        # Guardar gráficos de convergencia
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Loss
        if self.losses:
            steps, losses = zip(*self.losses)
            axes[0, 0].plot(steps, losses)
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_ylabel('Loss')
            
        # Gradients
        if self.gradients:
            steps, grads = zip(*self.gradients)
            axes[0, 1].plot(steps, grads)
            axes[0, 1].set_title('Gradient Norm')
            axes[0, 1].set_ylabel('Gradient Norm')
            
        # Learning Rate
        if self.learning_rates:
            steps, lrs = zip(*self.learning_rates)
            axes[0, 2].plot(steps, lrs)
            axes[0, 2].set_title('Learning Rate')
            axes[0, 2].set_ylabel('LR')
            
        # PSNR
        if self.psnr_values:
            steps, psnr = zip(*self.psnr_values)
            axes[1, 0].plot(steps, psnr)
            axes[1, 0].set_title('PSNR')
            axes[1, 0].set_ylabel('PSNR')
            
        # SSIM
        if self.ssim_values:
            steps, ssim = zip(*self.ssim_values)
            axes[1, 1].plot(steps, ssim)
            axes[1, 1].set_title('SSIM')
            axes[1, 1].set_ylabel('SSIM')
            
        # Binary Accuracy
        if self.image_quality:
            steps, acc, _ = zip(*self.image_quality)
            axes[1, 2].plot(steps, acc)
            axes[1, 2].set_title('Binary Accuracy')
            axes[1, 2].set_ylabel('Accuracy')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'convergence_plots.png'))
        plt.close()
        
    def save_logs(self):
        # Guardar logs en formato CSV
        import pandas as pd
        
        data = {
            'step': [],
            'loss': [],
            'gradient_norm': [],
            'learning_rate': [],
            'psnr': [],
            'ssim': [],
            'binary_accuracy': [],
            'edge_consistency': []
        }
        
        # Agregar datos
        for step, loss in self.losses:
            data['step'].append(step)
            data['loss'].append(loss)
            
        for step, grad in self.gradients:
            if step in data['step']:
                idx = data['step'].index(step)
                data['gradient_norm'].append(grad)
            else:
                data['step'].append(step)
                data['loss'].append(None)
                data['gradient_norm'].append(grad)
                
        for step, lr in self.learning_rates:
            if step in data['step']:
                idx = data['step'].index(step)
                data['learning_rate'].append(lr)
            else:
                data['step'].append(step)
                data['loss'].append(None)
                data['gradient_norm'].append(None)
                data['learning_rate'].append(lr)
                
        for step, psnr in self.psnr_values:
            if step in data['step']:
                idx = data['step'].index(step)
                data['psnr'].append(psnr)
            else:
                data['step'].append(step)
                data['loss'].append(None)
                data['gradient_norm'].append(None)
                data['learning_rate'].append(None)
                data['psnr'].append(psnr)
                
        for step, ssim in self.ssim_values:
            if step in data['step']:
                idx = data['step'].index(step)
                data['ssim'].append(ssim)
            else:
                data['step'].append(step)
                data['loss'].append(None)
                data['gradient_norm'].append(None)
                data['learning_rate'].append(None)
                data['psnr'].append(None)
                data['ssim'].append(ssim)
                
        for step, acc, edge in self.image_quality:
            if step in data['step']:
                idx = data['step'].index(step)
                data['binary_accuracy'].append(acc)
                data['edge_consistency'].append(edge)
            else:
                data['step'].append(step)
                data['loss'].append(None)
                data['gradient_norm'].append(None)
                data['learning_rate'].append(None)
                data['psnr'].append(None)
                data['ssim'].append(None)
                data['binary_accuracy'].append(acc)
                data['edge_consistency'].append(edge)
        
        # Rellenar valores faltantes
        max_len = len(data['step'])
        for key in data:
            while len(data[key]) < max_len:
                data[key].append(None)
        
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.log_dir, 'training_logs.csv'), index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='config/sketchy_binary_optimized.json', help='Path to option JSON file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # Create directories
    for key, path in opt['path'].items():
        if path is not None:
            os.makedirs(path, exist_ok=True)

    # Create logger
    logger = get_root_logger()
    logger.info('===> Initializing')
    logger.info(option.dict2str(opt))

    # Initialize convergence monitor
    monitor = ConvergenceMonitor(opt['path']['log'])

    # Create model
    model = create_model(opt)
    logger.info('===> Building model')

    # Create dataset and dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt, phase)
            train_loader = create_dataloader(train_set, dataset_opt, phase)
            logger.info('Number of train images: {:,d}'.format(len(train_set)))
            logger.info('Number of train batches: {:,d}'.format(len(train_loader)))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt, phase)
            val_loader = create_dataloader(val_set, dataset_opt, phase)
            logger.info('Number of val images: {:,d}'.format(len(val_set)))
            logger.info('Number of val batches: {:,d}'.format(len(val_loader)))

    # Create logger
    logger = Logger(opt)
    logger.info('===> Training')

    # Training
    current_step = 0
    current_epoch = 0
    n_iter = opt['train']['n_iter']
    val_freq = opt['train']['val_freq']
    save_checkpoint_freq = opt['train']['save_checkpoint_freq']
    print_freq = opt['train']['print_freq']

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.netG.parameters(),
        lr=opt['train']['optimizer']['lr'],
        weight_decay=opt['train']['optimizer'].get('weight_decay', 0)
    )

    # Initialize scheduler
    if 'scheduler' in opt['train']:
        if opt['train']['scheduler']['type'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=n_iter,
                eta_min=1e-6
            )
        else:
            scheduler = None
    else:
        scheduler = None

    # Initialize wandb if available
    if 'wandb' in opt and opt.get('log_wandb', False):
        wandb.init(project=opt['wandb']['project'], config=opt)

    # Training loop
    model.netG.train()
    for current_step in range(1, n_iter + 1):
        # Get batch
        for _, train_data in enumerate(train_loader):
            model.feed_data(train_data)
            
            # Forward pass
            loss = model.optimize_parameters()
            
            # Gradient clipping
            if opt['train'].get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.netG.parameters(), opt['train']['gradient_clip'])
            
            # Log loss
            monitor.log_loss(loss, current_step)
            
            # Log gradients
            monitor.log_gradients(model.netG, current_step)
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
                
            # Log learning rate
            monitor.log_lr(optimizer.param_groups[0]['lr'], current_step)
            
            # Debug: Log detailed information
            if args.debug and current_step % 10 == 0:
                logger.info(f'=== DEBUG STEP {current_step} ===')
                logger.info(f'Loss: {loss:.6f}')
                logger.info(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
                
                # Log data statistics
                for key, value in train_data.items():
                    if isinstance(value, torch.Tensor):
                        logger.info(f'{key} - Shape: {value.shape}, Min: {value.min():.4f}, Max: {value.max():.4f}, Mean: {value.mean():.4f}')
            
            # Log
            if current_step % print_freq == 0:
                logger.info('Iter: [{:d}/{:d}] Loss: {:.4f} LR: {:.6f}'.format(
                    current_step, n_iter, loss, optimizer.param_groups[0]['lr']))
                
                # Log to wandb
                if 'wandb' in opt and opt.get('log_wandb', False):
                    wandb.log({
                        'train/loss': loss,
                        'train/lr': optimizer.param_groups[0]['lr'],
                        'train/step': current_step
                    })
            
            # Validation
            if current_step % val_freq == 0:
                model.netG.eval()
                with torch.no_grad():
                    for _, val_data in enumerate(val_loader):
                        model.feed_data(val_data)
                        model.test()
                        visuals = model.get_current_visuals()
                        
                        # Save images
                        from core.metrics import tensor2img
                        sr_img = tensor2img(visuals['SR'][-1])
                        hr_img = tensor2img(visuals['HR'][-1])
                        lr_img = tensor2img(visuals['LR'][-1])
                        fake_img = tensor2img(visuals['INF'][-1])
                        
                        # Calculate metrics
                        psnr = calculate_psnr(sr_img, hr_img)
                        ssim = calculate_ssim(sr_img, hr_img)
                        
                        # Log metrics
                        monitor.log_metrics(psnr, ssim, current_step)
                        monitor.log_image_quality(sr_img, hr_img, current_step)
                        
                        logger.info('Validation - PSNR: {:.4f}, SSIM: {:.4f} (Best: {:.4f} at step {})'.format(
                            psnr, ssim, monitor.best_psnr, monitor.best_step))
                        
                        # Early stopping check
                        if monitor.should_stop():
                            logger.info(f'Early stopping triggered! No improvement for {monitor.patience} steps.')
                            logger.info(f'Best PSNR: {monitor.best_psnr:.4f} at step {monitor.best_step}')
                            return
                        
                        # Log to wandb
                        if 'wandb' in opt and opt.get('log_wandb', False):
                            wandb.log({
                                'val/psnr': psnr,
                                'val/ssim': ssim,
                                'val/step': current_step
                            })
                        break
                model.netG.train()
            
            # Save checkpoint and plots
            if current_step % save_checkpoint_freq == 0:
                logger.info('Saving models and training states.')
                model.save_network(current_epoch, current_step)
                
                # Save convergence plots and logs
                monitor.save_plots()
                monitor.save_logs()
            
            break  # Only one batch per iteration

    logger.info('===> Training finished')
    
    # Final save
    monitor.save_plots()
    monitor.save_logs()

if __name__ == '__main__':
    main() 