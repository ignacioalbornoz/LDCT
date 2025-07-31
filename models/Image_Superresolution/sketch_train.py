#!/usr/bin/env python3
import argparse
import logging
import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import options.options as option
from utils import util
from data import create_dataset, create_dataloader
from models import create_model
from utils.logger import Logger
from utils.wandb_logger import WandbLogger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/sketchy_edges2sketch_same_res.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
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
            train_set = create_dataset(dataset_opt, phase)
            train_loader = create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = create_dataset(dataset_opt, phase)
            val_loader = create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = create_model(opt)
    logger.info('Initial Model Finished')

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
                        diffusion.test(continous=True)
                        visuals = diffusion.get_current_visuals()

                        hr_img = util.tensor2img(visuals['HR'])  # uint8
                        lr_img = util.tensor2img(visuals['LR'])  # uint8
                        fake_img = util.tensor2img(visuals['INF'])  # uint8

                        # generation
                        util.save_img(
                            hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
                        util.save_img(
                            lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
                        util.save_img(
                            fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))
                        
                        # Use only the final generated image, not the grid
                        debug_logger.debug("=== BEFORE CONVERSION ===")
                        debug_logger.debug(f"visuals['SR'] type: {type(visuals['SR'])}")
                        debug_logger.debug(f"visuals['SR'] shape: {visuals['SR'].shape}")
                        debug_logger.debug(f"visuals['SR'][-1] shape: {visuals['SR'][-1].shape}")
                        
                        sr_img_final = util.tensor2img(visuals['SR'][-1])  # uint8
                        
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
                        avg_psnr += util.calculate_psnr(
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

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')
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

            hr_img = util.tensor2img(visuals['HR'])  # uint8
            lr_img = util.tensor2img(visuals['LR'])  # uint8
            fake_img = util.tensor2img(visuals['INF'])  # uint8
            
            debug_logger.debug("=== OTHER IMAGES ===")
            debug_logger.debug(f"hr_img shape: {hr_img.shape}, dtype: {hr_img.dtype}")
            debug_logger.debug(f"lr_img shape: {lr_img.shape}, dtype: {lr_img.dtype}")
            debug_logger.debug(f"fake_img shape: {fake_img.shape}, dtype: {fake_img.dtype}")

            sr_img_mode = 'grid'
            if sr_img_mode == 'single':
                # single img series
                sr_img = visuals['SR']  # uint8
                sample_num = sr_img.shape[0]
                for iter in range(0, sample_num):
                    util.save_img(
                        util.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
            else:
                # grid img
                sr_img = util.tensor2img(visuals['SR'])  # uint8
                util.save_img(
                    sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
                util.save_img(
                    util.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

            util.save_img(
                hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
            util.save_img(
                lr_img, '{}/{}_{}_lr.png'.format(result_path, current_step, idx))
            util.save_img(
                fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

            # generation
            eval_psnr = util.calculate_psnr(util.tensor2img(visuals['SR'][-1]), hr_img)
            eval_ssim = util.calculate_ssim(util.tensor2img(visuals['SR'][-1]), hr_img)

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(fake_img, util.tensor2img(visuals['SR'][-1]), hr_img, eval_psnr, eval_ssim)

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