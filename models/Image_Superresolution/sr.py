import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import wandb
import pandas as pd
from scipy import ndimage

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
        self.patience = 1000  # steps without improvement (mucho más paciente)
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
        # Solo activar early stopping después de al menos 1000 pasos
        min_steps = 1000
        if len(self.losses) < min_steps:
            return False
        return self.patience_counter >= self.patience
        
    def log_image_quality(self, sr_img, hr_img, step):
        # Calcular métricas de calidad de imagen
        binary_accuracy = np.mean((sr_img > 0.5) == (hr_img > 0.5))
        edge_consistency = self.calculate_edge_consistency(sr_img, hr_img)
        self.image_quality.append((step, binary_accuracy, edge_consistency))
        
    def calculate_edge_consistency(self, sr_img, hr_img):
        # Calcular consistencia de edges usando detección de bordes
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
        # Guardar logs en formato CSV de manera más simple
        try:
            # Crear un DataFrame más simple
            all_steps = set()
            
            # Recolectar todos los steps
            for step, _ in self.losses:
                all_steps.add(step)
            for step, _ in self.gradients:
                all_steps.add(step)
            for step, _ in self.learning_rates:
                all_steps.add(step)
            for step, _ in self.psnr_values:
                all_steps.add(step)
            for step, _ in self.ssim_values:
                all_steps.add(step)
            for step, _, _ in self.image_quality:
                all_steps.add(step)
            
            all_steps = sorted(list(all_steps))
            
            # Crear diccionarios para mapear step -> valor
            loss_dict = dict(self.losses)
            grad_dict = dict(self.gradients)
            lr_dict = dict(self.learning_rates)
            psnr_dict = dict(self.psnr_values)
            ssim_dict = dict(self.ssim_values)
            quality_dict = {step: (acc, edge) for step, acc, edge in self.image_quality}
            
            # Crear DataFrame
            data = []
            for step in all_steps:
                row = {
                    'step': step,
                    'loss': loss_dict.get(step),
                    'gradient_norm': grad_dict.get(step),
                    'learning_rate': lr_dict.get(step),
                    'psnr': psnr_dict.get(step),
                    'ssim': ssim_dict.get(step),
                    'binary_accuracy': quality_dict.get(step, (None, None))[0] if step in quality_dict else None,
                    'edge_consistency': quality_dict.get(step, (None, None))[1] if step in quality_dict else None
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(self.log_dir, 'training_logs.csv'), index=False)
            
        except Exception as e:
            print(f"Error saving logs: {e}")
            # Si falla, guardar solo los datos básicos
            if self.losses:
                basic_data = {'step': [step for step, _ in self.losses], 
                             'loss': [loss for _, loss in self.losses]}
                df = pd.DataFrame(basic_data)
                df.to_csv(os.path.join(self.log_dir, 'basic_training_logs.csv'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sketchy_binary_optimized.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('--debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])
    
    # Initialize convergence monitor
    monitor = ConvergenceMonitor(opt['path']['log'])
    
    # Setup debug logger for image shape debugging
    debug_logger = logging.getLogger('debug')
    debug_logger.setLevel(logging.DEBUG)
    debug_handler = logging.FileHandler(os.path.join(opt['path']['log'], 'debug_images.txt'))
    debug_handler.setLevel(logging.DEBUG)
    debug_formatter = logging.Formatter('%(asctime)s - %(message)s')
    debug_handler.setFormatter(debug_formatter)
    debug_logger.addHandler(debug_handler)
    debug_logger.propagate = False  # Don't propagate to console

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')
    
    # Initialize optimizer and scheduler
    optimizer = diffusion.optG
    if 'scheduler' in opt['train'] and opt['train']['scheduler']['type'] == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=opt['train']['n_iter'],
            eta_min=1e-6
        )
    else:
        scheduler = None

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            for _, train_data in enumerate(train_loader):
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                diffusion.optimize_parameters()
                
                # Gradient clipping
                if opt['train'].get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(diffusion.netG.parameters(), opt['train']['gradient_clip'])
                
                # Log loss and gradients for convergence monitoring
                logs = diffusion.get_current_log()
                if 'l_pix' in logs:
                    monitor.log_loss(logs['l_pix'], current_step)
                monitor.log_gradients(diffusion.netG, current_step)
                
                # Update learning rate scheduler
                if scheduler is not None:
                    scheduler.step()
                
                # Debug: Log detailed information
                if args.debug and current_step % 10 == 0:
                    logger.info(f'=== DEBUG STEP {current_step} ===')
                    logger.info(f'Loss: {logs.get("l_pix", "N/A")}')
                    logger.info(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
                    
                    # Log data statistics
                    for key, value in train_data.items():
                        if isinstance(value, torch.Tensor):
                            logger.info(f'{key} - Shape: {value.shape}, Min: {value.min():.4f}, Max: {value.max():.4f}, Mean: {value.mean():.4f}')
                
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8,d}> '.format(
                        current_epoch, current_step)
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['val'], schedule_phase='val')
                    for _,  val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()
                        sr_img = Metrics.tensor2img(visuals['SR'][-1])  # uint8 - Solo el último elemento del batch
                        hr_img = Metrics.tensor2img(visuals['HR'][-1])  # uint8 - Solo el último elemento del batch
                        lr_img = Metrics.tensor2img(visuals['LR'][-1])  # uint8 - Solo el último elemento del batch
                        fake_img = Metrics.tensor2img(visuals['INF'][-1])  # uint8 - Solo el último elemento del batch

                        # Calculate metrics
                        psnr = Metrics.calculate_psnr(sr_img, hr_img)
                        ssim = Metrics.calculate_ssim(sr_img, hr_img)
                        
                        # Log metrics for convergence monitoring
                        monitor.log_metrics(psnr, ssim, current_step)
                        monitor.log_image_quality(sr_img, hr_img, current_step)
                        
                        # Early stopping check
                        if monitor.should_stop():
                            logger.info(f'Early stopping triggered! No improvement for {monitor.patience} steps.')
                            logger.info(f'Best PSNR: {monitor.best_psnr:.4f} at step {monitor.best_step}')
                            break
                        
                        # generation
                        Metrics.save_img(
                            hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            sr_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))
                        # Use only the final generated image, not the grid
                        debug_logger.debug("=== BEFORE CONVERSION ===")
                        debug_logger.debug(f"visuals['SR'] type: {type(visuals['SR'])}")
                        debug_logger.debug(f"visuals['SR'] shape: {visuals['SR'].shape}")
                        debug_logger.debug(f"visuals['SR'][-1] shape: {visuals['SR'][-1].shape}")
                        
                        sr_img_final = Metrics.tensor2img(visuals['SR'][-1])  # uint8
                        
                        debug_logger.debug("=== AFTER CONVERSION ===")
                        debug_logger.debug(f"sr_img_final shape: {sr_img_final.shape}")
                        debug_logger.debug(f"sr_img_final dtype: {sr_img_final.dtype}")
                        
                        # Ensure all images have 3 dimensions for concatenation (for visualization)
                        if sr_img_final.ndim == 2:
                            sr_img_final = np.expand_dims(sr_img_final, axis=2)
                        if fake_img.ndim == 2:
                            fake_img = np.expand_dims(fake_img, axis=2)
                        if hr_img.ndim == 2:
                            hr_img = np.expand_dims(hr_img, axis=2)
                            
                        # Ensure all images have the same shape for concatenation
                        if sr_img_final.shape != fake_img.shape or sr_img_final.shape != hr_img.shape:
                            # Resize to match the smallest dimension
                            min_height = min(sr_img_final.shape[0], fake_img.shape[0], hr_img.shape[0])
                            min_width = min(sr_img_final.shape[1], fake_img.shape[1], hr_img.shape[1])
                            
                            if sr_img_final.shape[:2] != (min_height, min_width):
                                sr_img_final = sr_img_final[:min_height, :min_width, :]
                            if fake_img.shape[:2] != (min_height, min_width):
                                fake_img = fake_img[:min_height, :min_width, :]
                            if hr_img.shape[:2] != (min_height, min_width):
                                hr_img = hr_img[:min_height, :min_width, :]
                        
                        # DEBUG: Log detailed information about all images
                        debug_logger.debug("=== DEBUG INFO ===")
                        debug_logger.debug(f"sr_img_final shape: {sr_img_final.shape}, dtype: {sr_img_final.dtype}")
                        debug_logger.debug(f"fake_img shape: {fake_img.shape}, dtype: {fake_img.dtype}")
                        debug_logger.debug(f"hr_img shape: {hr_img.shape}, dtype: {hr_img.dtype}")
                        debug_logger.debug(f"sr_img_final min/max: {sr_img_final.min()}/{sr_img_final.max()}")
                        debug_logger.debug(f"fake_img min/max: {fake_img.min()}/{fake_img.max()}")
                        debug_logger.debug(f"hr_img min/max: {hr_img.min()}/{hr_img.max()}")
                        
                        # Force all images to be grayscale (1 channel) for consistent concatenation
                        # This is a temporary fix until we figure out why some images have different channels
                        if sr_img_final.shape[2] > 1:
                            debug_logger.debug(f"Converting sr_img_final from {sr_img_final.shape[2]} channels to 1")
                            sr_img_final = np.mean(sr_img_final, axis=2, keepdims=True).astype(sr_img_final.dtype)
                        if fake_img.shape[2] > 1:
                            debug_logger.debug(f"Converting fake_img from {fake_img.shape[2]} channels to 1")
                            fake_img = np.mean(fake_img, axis=2, keepdims=True).astype(fake_img.dtype)
                        if hr_img.shape[2] > 1:
                            debug_logger.debug(f"Converting hr_img from {hr_img.shape[2]} channels to 1")
                            hr_img = np.mean(hr_img, axis=2, keepdims=True).astype(hr_img.dtype)
                        
                        debug_logger.debug("=== AFTER CONVERSION ===")
                        debug_logger.debug(f"sr_img_final shape: {sr_img_final.shape}")
                        debug_logger.debug(f"fake_img shape: {fake_img.shape}")
                        debug_logger.debug(f"hr_img shape: {hr_img.shape}")
                            
                        # Final verification before concatenation
                        debug_logger.debug("=== FINAL VERIFICATION ===")
                        debug_logger.debug(f"All shapes should be identical:")
                        debug_logger.debug(f"fake_img: {fake_img.shape}")
                        debug_logger.debug(f"sr_img_final: {sr_img_final.shape}")
                        debug_logger.debug(f"hr_img: {hr_img.shape}")
                        
                        # Verify all shapes are the same
                        if fake_img.shape != sr_img_final.shape or sr_img_final.shape != hr_img.shape:
                            debug_logger.error(f"ERROR: Shapes don't match!")
                            debug_logger.error(f"fake_img: {fake_img.shape}")
                            debug_logger.error(f"sr_img_final: {sr_img_final.shape}")
                            debug_logger.error(f"hr_img: {hr_img.shape}")
                            raise ValueError("Image shapes don't match for concatenation")
                        
                        tb_logger.add_image(
                            'Iter_{}'.format(current_step),
                            np.transpose(np.concatenate(
                                (fake_img, sr_img_final, hr_img), axis=1), [2, 0, 1]),
                            idx)
                        avg_psnr += Metrics.calculate_psnr(
                            sr_img_final, hr_img)

                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}', 
                                np.concatenate((fake_img, sr_img_final, hr_img), axis=1)
                            )

                    avg_psnr = avg_psnr / idx
                    diffusion.set_new_noise_schedule(
                        opt['model']['beta_schedule']['train'], schedule_phase='train')
                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                        current_epoch, current_step, avg_psnr))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_step': val_step
                        })
                        val_step += 1

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)
                    
                    # Save convergence plots and logs
                    monitor.save_plots()
                    monitor.save_logs()

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')
        
        # Final save of convergence data
        monitor.save_plots()
        monitor.save_logs()
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)
        for _,  val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=True)
            visuals = diffusion.get_current_visuals()

            debug_logger.debug("=== BEFORE TENSOR2IMG ===")
            debug_logger.debug(f"visuals['HR'] type: {type(visuals['HR'])}")
            debug_logger.debug(f"visuals['HR'] shape: {visuals['HR'].shape}")
            debug_logger.debug(f"visuals['HR'] dtype: {visuals['HR'].dtype}")
            debug_logger.debug(f"visuals['HR'] min/max: {visuals['HR'].min()}/{visuals['HR'].max()}")
            
            hr_img = Metrics.tensor2img(visuals['HR'][-1])  # uint8 - Solo el último elemento del batch
            lr_img = Metrics.tensor2img(visuals['LR'][-1])  # uint8 - Solo el último elemento del batch
            fake_img = Metrics.tensor2img(visuals['INF'][-1])  # uint8 - Solo el último elemento del batch
            
            debug_logger.debug("=== AFTER TENSOR2IMG ===")
            debug_logger.debug(f"hr_img shape: {hr_img.shape}, dtype: {hr_img.dtype}")
            debug_logger.debug(f"lr_img shape: {lr_img.shape}, dtype: {lr_img.dtype}")
            debug_logger.debug(f"fake_img shape: {fake_img.shape}, dtype: {fake_img.dtype}")

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                sr_img = visuals['SR']  # uint8
                sample_num = sr_img.shape[0]
                for iter in range(0, sample_num):
                    Metrics.save_img(
                        Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
            else:
                # grid img
                sr_img = Metrics.tensor2img(visuals['SR'][-1])  # uint8 - Solo el último elemento del batch
                Metrics.save_img(
                    sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
                Metrics.save_img(
                    Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

            Metrics.save_img(
                hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

            # generation
            eval_psnr = Metrics.calculate_psnr(Metrics.tensor2img(visuals['SR'][-1]), hr_img)
            eval_ssim = Metrics.calculate_ssim(Metrics.tensor2img(visuals['SR'][-1]), hr_img)

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(fake_img, Metrics.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr, eval_ssim)

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx

        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim：{:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim)
            })
