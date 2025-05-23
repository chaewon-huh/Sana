# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import gc
import getpass
import hashlib
import json
import os
import os.path as osp
import time
import warnings
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

warnings.filterwarnings("ignore")  # ignore warning

import numpy as np
import pyrallis
import torch
from accelerate import Accelerator, InitProcessGroupKwargs, skip_first_batches
from PIL import Image
from termcolor import colored

from diffusion import DPMS, FlowEuler, Scheduler
from diffusion.data.builder import build_dataloader, build_dataset
from diffusion.data.wids import DistributedRangedSampler
from diffusion.model.builder import build_model, get_vae, vae_decode, vae_encode
from diffusion.model.model_growth_utils import ModelGrowthInitializer
from diffusion.model.respace import compute_density_for_timestep_sampling
from diffusion.model.utils import get_weight_dtype
from diffusion.utils.checkpoint import load_checkpoint, save_checkpoint
from diffusion.utils.config import SanaConfig, model_init_config
from diffusion.utils.data_sampler import AspectRatioBatchSampler
from diffusion.utils.dist_utils import flush, get_world_size
from diffusion.utils.logger import LogBuffer, get_root_logger
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import DebugUnderflowOverflow, init_random_seed, set_random_seed
from diffusion.utils.optimizer import auto_scale_lr, build_optimizer
from diffusion.model.nets.sana_blocks import ImageConditionEmbedder

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_fsdp_env():
    # Basic FSDP settings
    os.environ["ACCELERATE_USE_FSDP"] = "true"

    # Auto wrapping policy
    os.environ["FSDP_AUTO_WRAP_POLICY"] = "TRANSFORMER_BASED_WRAP"
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "SanaMSBlock"  # Your transformer block name

    # Performance optimization settings
    os.environ["FSDP_BACKWARD_PREFETCH"] = "BACKWARD_PRE"
    os.environ["FSDP_FORWARD_PREFETCH"] = "false"

    # State dict settings
    os.environ["FSDP_STATE_DICT_TYPE"] = "FULL_STATE_DICT"
    os.environ["FSDP_SYNC_MODULE_STATES"] = "true"
    os.environ["FSDP_USE_ORIG_PARAMS"] = "true"

    # Sharding strategy
    os.environ["FSDP_SHARDING_STRATEGY"] = "FULL_SHARD"

    # Memory optimization settings (optional)
    os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "false"
    os.environ["FSDP_OFFLOAD_PARAMS"] = "false"

    # Precision settings
    os.environ["FSDP_REDUCE_SCATTER_PRECISION"] = "fp32"
    os.environ["FSDP_ALL_GATHER_PRECISION"] = "fp32"
    os.environ["FSDP_OPTIMIZER_STATE_PRECISION"] = "fp32"


def ema_update(model_dest, model_src, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


@torch.inference_mode()
def log_validation(accelerator, config, model, logger, step, device, vae=None, init_noise=None):
    torch.cuda.empty_cache()
    vis_sampler = config.scheduler.vis_sampler
    model = accelerator.unwrap_model(model).eval()
    hw = torch.tensor([[image_size, image_size]], dtype=torch.float, device=device).repeat(1, 1)
    ar = torch.tensor([[1.0]], device=device).repeat(1, 1)
    
    # Create null condition (zeros for image condition)
    null_condition = torch.zeros(1, config.vae.vae_latent_dim, latent_size, latent_size, device=device)

    # Create sampling noise:
    logger.info("Running validation... ")
    image_logs = []

    def run_sampling(init_z=None, label_suffix="", vae=None, sampler="dpm-solver", condition_latent=None):
        latents = []
        current_image_logs = []
        
        # Use condition_latent if provided, otherwise use null condition
        if condition_latent is None:
            condition_latent = null_condition
            
        for i in range(len(validation_source_images)):
            z = (
                torch.randn(1, config.vae.vae_latent_dim, latent_size, latent_size, device=device)
                if init_z is None
                else init_z
            )
            
            # Load and encode source image as condition
            if i < len(validation_source_images):
                source_img_path = validation_source_images[i]
                source_img = Image.open(source_img_path).convert("RGB")
                source_img = source_img.resize((image_size, image_size))
                source_tensor = torch.tensor(np.array(source_img)).permute(2, 0, 1).float() / 255.0
                source_tensor = source_tensor.unsqueeze(0).to(device)
                source_tensor = (source_tensor - 0.5) * 2  # normalize to [-1, 1]
                
                with torch.no_grad():
                    condition_latent = vae_encode(config.vae.vae_type, vae, source_tensor, False, device).to(device)
            
            model_kwargs = dict(data_info={"img_hw": hw, "aspect_ratio": ar}, condition_latent=condition_latent)

            if sampler == "dpm-solver":
                dpm_solver = DPMS(
                    model.forward_with_dpmsolver,
                    condition=condition_latent,
                    uncondition=null_condition,
                    cfg_scale=1.0,  # No CFG for image condition
                    model_kwargs=model_kwargs,
                )
                denoised = dpm_solver.sample(
                    z,
                    steps=14,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep",
                )
            elif sampler == "flow_euler":
                flow_solver = FlowEuler(
                    model, condition=condition_latent, uncondition=null_condition, cfg_scale=1.0, model_kwargs=model_kwargs
                )
                denoised = flow_solver.sample(z, steps=28)
            elif sampler == "flow_dpm-solver":
                dpm_solver = DPMS(
                    model.forward_with_dpmsolver,
                    condition=condition_latent,
                    uncondition=null_condition,
                    cfg_scale=1.0,
                    model_type="flow",
                    model_kwargs=model_kwargs,
                    schedule="FLOW",
                )
                denoised = dpm_solver.sample(
                    z,
                    steps=20,
                    order=2,
                    skip_type="time_uniform_flow",
                    method="multistep",
                    flow_shift=config.scheduler.flow_shift,
                )
            else:
                raise ValueError(f"{sampler} not implemented")

            latents.append(denoised)
        torch.cuda.empty_cache()
        if vae is None:
            vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, accelerator.device).to(vae_dtype)
        for i, latent in enumerate(latents):
            latent = latent.to(vae_dtype)
            samples = vae_decode(config.vae.vae_type, vae, latent)
            samples = (
                torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
            )
            image = Image.fromarray(samples)
            current_image_logs.append({"validation_prompt": f"img2img_{i}{label_suffix}", "images": [image]})

        return current_image_logs

    # First run with original noise
    image_logs += run_sampling(init_z=None, label_suffix="", vae=vae, sampler=vis_sampler)

    # Second run with init_noise if provided
    if init_noise is not None:
        torch.cuda.empty_cache()
        gc.collect()
        init_noise = torch.clone(init_noise).to(device)
        image_logs += run_sampling(init_z=init_noise, label_suffix=" w/ init noise", vae=vae, sampler=vis_sampler)

    formatted_images = []
    for log in image_logs:
        images = log["images"]
        validation_prompt = log["validation_prompt"]
        for image in images:
            formatted_images.append((validation_prompt, np.asarray(image)))

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for validation_prompt, image in formatted_images:
                tracker.writer.add_images(validation_prompt, image[None, ...], step, dataformats="NHWC")
        elif tracker.name == "wandb":
            import wandb

            wandb_images = []
            for validation_prompt, image in formatted_images:
                wandb_images.append(wandb.Image(image, caption=validation_prompt, file_type="jpg"))
            tracker.log({"validation": wandb_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    def concatenate_images(image_caption, images_per_row=5, image_format="webp"):
        import io

        images = [log["images"][0] for log in image_caption]
        if images[0].size[0] > 1024:
            images = [image.resize((1024, 1024)) for image in images]

        widths, heights = zip(*(img.size for img in images))
        max_width = max(widths)
        total_height = sum(heights[i : i + images_per_row][0] for i in range(0, len(images), images_per_row))

        new_im = Image.new("RGB", (max_width * images_per_row, total_height))

        y_offset = 0
        for i in range(0, len(images), images_per_row):
            row_images = images[i : i + images_per_row]
            x_offset = 0
            for img in row_images:
                new_im.paste(img, (x_offset, y_offset))
                x_offset += max_width
            y_offset += heights[i]
        webp_image_bytes = io.BytesIO()
        new_im.save(webp_image_bytes, format=image_format)
        webp_image_bytes.seek(0)
        new_im = Image.open(webp_image_bytes)

        return new_im

    if config.train.local_save_vis:
        file_format = "webp"
        local_vis_save_path = osp.join(config.work_dir, "log_vis")
        os.umask(0o000)
        os.makedirs(local_vis_save_path, exist_ok=True)
        concatenated_image = concatenate_images(image_logs, images_per_row=5, image_format=file_format)
        save_path = (
            osp.join(local_vis_save_path, f"vis_{step}.{file_format}")
            if init_noise is None
            else osp.join(local_vis_save_path, f"vis_{step}_w_init.{file_format}")
        )
        concatenated_image.save(save_path)

    model.train()
    del vae
    flush()
    return image_logs


class Img2ImgModelWrapper(torch.nn.Module):
    """Wrapper for SanaMS model to handle image-to-image conditioning"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, *args, **kwargs):
        # Check if img_condition is provided
        if 'img_condition' in kwargs:
            img_condition = kwargs.pop('img_condition')
            # Use the image condition directly with the new ImageConditionEmbedder
            # The y_embedder has been replaced with ImageConditionEmbedder
            # Convert the image condition to the expected format
            y_processed = self.model.y_embedder(img_condition, self.model.training)
            kwargs['y'] = y_processed
            
            # Create a simple mask for the image tokens
            B, seq_len, _ = y_processed.shape  
            mask = torch.ones(B, 1, 1, seq_len, device=y_processed.device, dtype=torch.bool)
            kwargs['mask'] = mask
            
        return self.model(*args, **kwargs)
        
    def __getattr__(self, name):
        # Delegate attribute access to wrapped model
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def train(
    config, args, accelerator, model, model_ema, optimizer, lr_scheduler, train_dataloader, train_diffusion, logger
):
    if getattr(config.train, "debug_nan", False):
        DebugUnderflowOverflow(model, max_frames_to_save=100)
        logger.info("NaN debugger registered. Start to detect overflow during training.")
    log_buffer = LogBuffer()

    global_step = start_step + 1
    skip_step = max(config.train.skip_step, global_step) % train_dataloader_len
    skip_step = skip_step if skip_step < (train_dataloader_len - 20) else 0
    loss_nan_timer = 0
    model_instance.to(accelerator.device)

    # Cache Dataset for BatchSampler
    if args.caching and config.model.multi_scale:
        caching_start = time.time()
        logger.info(
            f"Start caching your dataset for batch_sampler at {cache_file}. \n"
            f"This may take a lot of time...No training will launch"
        )
        train_dataloader.batch_sampler.sampler.set_start(max(train_dataloader.batch_sampler.exist_ids, 0))
        accelerator.wait_for_everyone()
        for index, _ in enumerate(train_dataloader):
            accelerator.wait_for_everyone()
            if index % 2000 == 0:
                logger.info(
                    f"rank: {rank}, Cached file len: {len(train_dataloader.batch_sampler.cached_idx)} / {len(train_dataloader)}"
                )
                print(
                    f"rank: {rank}, Cached file len: {len(train_dataloader.batch_sampler.cached_idx)} / {len(train_dataloader)}"
                )
            if (time.time() - caching_start) / 3600 > 3.7:
                json.dump(train_dataloader.batch_sampler.cached_idx, open(cache_file, "w"), indent=4)
                accelerator.wait_for_everyone()
                break
            if len(train_dataloader.batch_sampler.cached_idx) == len(train_dataloader) - 1000:
                logger.info(
                    f"Saving rank: {rank}, Cached file len: {len(train_dataloader.batch_sampler.cached_idx)} / {len(train_dataloader)}"
                )
                json.dump(train_dataloader.batch_sampler.cached_idx, open(cache_file, "w"), indent=4)
            accelerator.wait_for_everyone()
            continue
        accelerator.wait_for_everyone()
        print(f"Saving rank-{rank} Cached file len: {len(train_dataloader.batch_sampler.cached_idx)}")
        json.dump(train_dataloader.batch_sampler.cached_idx, open(cache_file, "w"), indent=4)
        return

    # Now you train the model
    for epoch in range(start_epoch + 1, config.train.num_epochs + 1):
        time_start, last_tic = time.time(), time.time()
        sampler = (
            train_dataloader.batch_sampler.sampler
            if (num_replicas > 1 or config.model.multi_scale)
            else train_dataloader.sampler
        )
        sampler.set_epoch(epoch)
        sampler.set_start(max((skip_step - 1) * config.train.train_batch_size, 0))
        if skip_step > 1 and accelerator.is_main_process:
            logger.info(f"Skipped Steps: {skip_step}")
        skip_step = 1
        data_time_start = time.time()
        data_time_all = 0
        vae_time_all = 0
        model_time_all = 0
        for step, batch in enumerate(train_dataloader):
            # batch contains: [source_image, target_image, data_info]
            accelerator.wait_for_everyone()
            data_time_all += time.time() - data_time_start
            vae_time_start = time.time()
            
            # Encode source image as condition
            if load_vae_feat:
                source_latent = batch[0].to(accelerator.device)  # pre-encoded source latent
                target_latent = batch[1].to(accelerator.device)  # pre-encoded target latent
            else:
                with torch.no_grad():
                    source_latent = vae_encode(config.vae.vae_type, vae, batch[0], config.vae.sample_posterior, accelerator.device)
                    target_latent = vae_encode(config.vae.vae_type, vae, batch[1], config.vae.sample_posterior, accelerator.device)

            accelerator.wait_for_everyone()
            vae_time_all += time.time() - vae_time_start

            clean_images = target_latent
            condition_latent = source_latent
            data_info = batch[2] if len(batch) > 2 else {}

            # Sample a random timestep for each image
            bs = clean_images.shape[0]
            timesteps = torch.randint(
                0, config.scheduler.train_sampling_steps, (bs,), device=clean_images.device
            ).long()
            if config.scheduler.weighting_scheme in ["logit_normal"]:
                # adapting from diffusers.training_utils
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=config.scheduler.weighting_scheme,
                    batch_size=bs,
                    logit_mean=config.scheduler.logit_mean,
                    logit_std=config.scheduler.logit_std,
                    mode_scale=None,  # not used
                )
                timesteps = (u * config.scheduler.train_sampling_steps).long().to(clean_images.device)
            grad_norm = None
            accelerator.wait_for_everyone()
            
            model_time_start = time.time()
            with accelerator.accumulate(model):
                # Predict the noise residual
                optimizer.zero_grad()
                
                # For image-to-image, pass the condition image directly
                # The ImageConditionEmbedder will handle the conversion
                
                loss_term = train_diffusion.training_losses(
                    model, clean_images, timesteps, 
                    model_kwargs=dict(
                        data_info=data_info,
                        img_condition=condition_latent,  # Pass image condition directly
                    )
                )

                loss = loss_term["loss"].mean()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.train.gradient_clip)
                    if not config.train.use_fsdp and config.train.ema_update and model_ema is not None:
                        ema_update(model_ema, model, config.train.ema_rate)

                optimizer.step()
                lr_scheduler.step()
                accelerator.wait_for_everyone()
                model_time_all += time.time() - model_time_start

            if torch.any(torch.isnan(loss)):
                loss_nan_timer += 1
            lr = lr_scheduler.get_last_lr()[0]
            logs = {args.loss_report_name: accelerator.gather(loss).mean().item()}
            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            log_buffer.update(logs)
            if (step + 1) % config.train.log_interval == 0 or (step + 1) == 1:
                accelerator.wait_for_everyone()
                t = (time.time() - last_tic) / config.train.log_interval
                t_d = data_time_all / config.train.log_interval
                t_m = model_time_all / config.train.log_interval
                t_vae = vae_time_all / config.train.log_interval
                avg_time = (time.time() - time_start) / (step + 1)
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - global_step - 1))))
                eta_epoch = str(
                    datetime.timedelta(
                        seconds=int(
                            avg_time
                            * (train_dataloader_len - sampler.step_start // config.train.train_batch_size - step - 1)
                        )
                    )
                )
                log_buffer.average()

                current_step = (
                    global_step - sampler.step_start // config.train.train_batch_size
                ) % train_dataloader_len
                current_step = train_dataloader_len if current_step == 0 else current_step

                info = (
                    f"Epoch: {epoch} | Global Step: {global_step} | Local Step: {current_step} // {train_dataloader_len}, "
                    f"total_eta: {eta}, epoch_eta:{eta_epoch}, time: all:{t:.3f}, model:{t_m:.3f}, data:{t_d:.3f}, "
                    f"vae:{t_vae:.3f}, lr:{lr:.3e}, "
                )
                info += (
                    f"s:({model.module.h}, {model.module.w}), "
                    if hasattr(model, "module")
                    else f"s:({model.h}, {model.w}), "
                )

                info += ", ".join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
                model_time_all = 0
                vae_time_all = 0
                if accelerator.is_main_process:
                    logger.info(info)

            logs.update(lr=lr)
            if accelerator.is_main_process:
                accelerator.log(logs, step=global_step)

            global_step += 1

            if loss_nan_timer > 20:
                raise ValueError("Loss is NaN too much times. Break here.")
            if (
                global_step % config.train.save_model_steps == 0
                or (time.time() - training_start_time) / 3600 > config.train.early_stop_hours
            ):
                torch.cuda.synchronize()
                accelerator.wait_for_everyone()

                # Choose different saving methods based on whether FSDP is used
                if config.train.use_fsdp:
                    # FSDP mode
                    os.umask(0o000)
                    ckpt_saved_path = save_checkpoint(
                        work_dir=osp.join(config.work_dir, "checkpoints"),
                        epoch=epoch,
                        model=model,
                        accelerator=accelerator,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        step=global_step,
                        add_symlink=True,
                    )
                else:
                    # DDP mode
                    if accelerator.is_main_process:
                        os.umask(0o000)
                        ckpt_saved_path = save_checkpoint(
                            work_dir=osp.join(config.work_dir, "checkpoints"),
                            epoch=epoch,
                            model=accelerator.unwrap_model(model),
                            model_ema=accelerator.unwrap_model(model_ema) if model_ema is not None else None,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            step=global_step,
                            generator=generator,
                            add_symlink=True,
                        )

                if accelerator.is_main_process:
                    if config.train.online_metric and global_step % config.train.eval_metric_step == 0 and step > 1:
                        online_metric_monitor_dir = osp.join(config.work_dir, config.train.online_metric_dir)
                        os.makedirs(online_metric_monitor_dir, exist_ok=True)
                        with open(f"{online_metric_monitor_dir}/{ckpt_saved_path.split('/')[-1]}.txt", "w") as f:
                            f.write(osp.join(config.work_dir, "config.py") + "\n")
                            f.write(ckpt_saved_path)

                if (time.time() - training_start_time) / 3600 > config.train.early_stop_hours:
                    logger.info(f"Stopping training at epoch {epoch}, step {global_step} due to time limit.")
                    return

            if config.train.visualize and (global_step % config.train.eval_sampling_steps == 0 or (step + 1) == 1):
                if config.train.use_fsdp:
                    merged_state_dict = accelerator.get_state_dict(model)

                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    if config.train.use_fsdp:
                        model_instance.load_state_dict(merged_state_dict)
                    if validation_noise is not None:
                        log_validation(
                            accelerator=accelerator,
                            config=config,
                            model=model_instance,
                            logger=logger,
                            step=global_step,
                            device=accelerator.device,
                            vae=vae,
                            init_noise=validation_noise,
                        )
                    else:
                        log_validation(
                            accelerator=accelerator,
                            config=config,
                            model=model_instance,
                            logger=logger,
                            step=global_step,
                            device=accelerator.device,
                            vae=vae,
                        )

            # avoid dead-lock of multiscale data batch sampler
            if (
                config.model.multi_scale
                and (train_dataloader_len - sampler.step_start // config.train.train_batch_size - step) < 30
            ):
                global_step = (
                    (global_step + train_dataloader_len - 1) // train_dataloader_len
                ) * train_dataloader_len + 1
                logger.info("Early stop current iteration")
                skip_first_batches(train_dataloader, True)
                break

            data_time_start = time.time()

        if epoch % config.train.save_model_epochs == 0 or epoch == config.train.num_epochs and not config.debug:
            accelerator.wait_for_everyone()
            torch.cuda.synchronize()

            # Choose different saving methods based on whether FSDP is used
            if config.train.use_fsdp:
                # FSDP mode
                os.umask(0o000)
                ckpt_saved_path = save_checkpoint(
                    work_dir=osp.join(config.work_dir, "checkpoints"),
                    epoch=epoch,
                    model=model,
                    accelerator=accelerator,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    step=global_step,
                    add_symlink=True,
                )
            else:
                # DDP mode
                if accelerator.is_main_process:
                    os.umask(0o000)
                    ckpt_saved_path = save_checkpoint(
                        osp.join(config.work_dir, "checkpoints"),
                        epoch=epoch,
                        step=global_step,
                        model=accelerator.unwrap_model(model),
                        model_ema=accelerator.unwrap_model(model_ema) if model_ema is not None else None,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        generator=generator,
                        add_symlink=True,
                    )

            if accelerator.is_main_process:
                online_metric_monitor_dir = osp.join(config.work_dir, config.train.online_metric_dir)
                os.makedirs(online_metric_monitor_dir, exist_ok=True)
                with open(f"{online_metric_monitor_dir}/{ckpt_saved_path.split('/')[-1]}.txt", "w") as f:
                    f.write(osp.join(config.work_dir, "config.py") + "\n")
                    f.write(ckpt_saved_path)


@pyrallis.wrap()
def main(cfg: SanaConfig) -> None:
    global train_dataloader_len, start_epoch, start_step, vae, generator, num_replicas, rank, training_start_time
    global load_vae_feat, validation_noise, validation_source_images
    global latent_size, image_size, cache_file, total_steps, vae_dtype, model_instance

    config = cfg
    args = cfg

    # 1.Initialize training mode
    if config.train.use_fsdp:
        set_fsdp_env()
        init_train = "FSDP"
    else:
        init_train = "DDP"

    training_start_time = time.time()
    load_from = True

    if args.resume_from or config.model.resume_from:
        load_from = False
        config.model.resume_from = dict(
            checkpoint=args.resume_from or config.model.resume_from,
            load_ema=False,
            resume_optimizer=True,
            resume_lr_scheduler=config.train.resume_lr_scheduler,
        )

    if args.debug:
        config.train.log_interval = 1
        config.train.train_batch_size = min(64, config.train.train_batch_size)
        args.report_to = "tensorboard"

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.model.mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=osp.join(config.work_dir, "logs"),
        kwargs_handlers=[init_handler],
    )

    log_name = "train_log.log"
    logger = get_root_logger(osp.join(config.work_dir, log_name))
    logger.info(accelerator.state)

    config.train.seed = init_random_seed(getattr(config.train, "seed", None))
    set_random_seed(config.train.seed + int(os.environ["LOCAL_RANK"]))
    generator = torch.Generator(device="cpu").manual_seed(config.train.seed)

    if accelerator.is_main_process:
        pyrallis.dump(config, open(osp.join(config.work_dir, "config.yaml"), "w"), sort_keys=False, indent=4)
        if args.report_to == "wandb":
            import wandb

            wandb.init(project=args.tracker_project_name, name=args.name, resume="allow", id=args.name)

    logger.info(f"Config: \n{config}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.train.seed}")
    logger.info(f"Initializing: {init_train} for Image-to-Image training")

    image_size = config.model.image_size
    latent_size = int(image_size) // config.vae.vae_downsample_rate
    pred_sigma = getattr(config.scheduler, "pred_sigma", True)
    learn_sigma = getattr(config.scheduler, "learn_sigma", True) and pred_sigma
    vae = None
    vae_dtype = get_weight_dtype(config.vae.weight_dtype)

    validation_noise = (
        torch.randn(1, config.vae.vae_latent_dim, latent_size, latent_size, device="cpu", generator=generator)
        if getattr(config.train, "deterministic_validation", False)
        else None
    )
    if not config.data.load_vae_feat:
        vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, accelerator.device).to(vae_dtype)

    logger.info(f"vae type: {config.vae.vae_type}, path: {config.vae.vae_pretrained}, weight_dtype: {vae_dtype}")

    # No text encoder needed for image-to-image
    text_embed_dim = 2304  # Keep original text embedding dimension to match checkpoint

    # Setup validation source images if provided
    validation_source_images = getattr(config.train, "validation_source_images", [])
    
    os.environ["AUTOCAST_LINEAR_ATTN"] = "true" if config.model.autocast_linear_attn else "false"

    # 3. build scheduler
    train_diffusion = Scheduler(
        str(config.scheduler.train_sampling_steps),
        noise_schedule=config.scheduler.noise_schedule,
        predict_flow_v=config.scheduler.predict_flow_v,
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        snr=config.train.snr_loss,
        flow_shift=config.scheduler.flow_shift,
    )
    predict_info = (
        f"flow-prediction: {config.scheduler.predict_flow_v}, noise schedule: {config.scheduler.noise_schedule}"
    )
    if "flow" in config.scheduler.noise_schedule:
        predict_info += f", flow shift: {config.scheduler.flow_shift}"
    if config.scheduler.weighting_scheme in ["logit_normal", "mode"]:
        predict_info += (
            f", flow weighting: {config.scheduler.weighting_scheme}, "
            f"logit-mean: {config.scheduler.logit_mean}, logit-std: {config.scheduler.logit_std}"
        )
    logger.info(predict_info)

    # 4. build models
    model_kwargs = model_init_config(config, latent_size=latent_size)
    # Keep original caption_channels to match checkpoint
    # model_kwargs['caption_channels'] = text_embed_dim  # Override caption channels for image condition
    
    # Create null embedding path for image-to-image training
    null_embed_path = "output/pretrained_models/null_embed_img2img.pth"
    
    model = build_model(
        config.model.model,
        config.train.grad_checkpointing,
        getattr(config.model, "fp32_attention", False),
        null_embed_path=null_embed_path,  # Use the dummy null embedding
        **model_kwargs,
    ).train()
    
    # Replace y_embedder with ImageConditionEmbedder for image conditioning
    logger.info("Replacing y_embedder with ImageConditionEmbedder for image-to-image training")
    vae_latent_channels = 32  # Standard VAE latent channels for SANA
    
    # Get model device and dtype
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    
    # Create new ImageConditionEmbedder
    img_embedder = ImageConditionEmbedder(
        in_channels=vae_latent_channels,
        hidden_size=model.hidden_size,
        uncond_prob=model.y_embedder.uncond_prob,
        token_num=getattr(model.y_embedder, 'token_num', 300)
    )
    
    # Replace the y_embedder
    model.y_embedder = img_embedder.to(model_device, model_dtype)
    
    logger.info(f"Replaced y_embedder with ImageConditionEmbedder")

    # Wrap model for image-to-image conditioning AFTER y_embedder replacement
    model = Img2ImgModelWrapper(model)

    if (not config.train.use_fsdp) and config.train.ema_update:
        model_ema = deepcopy(model).eval()
        logger.info("Creating EMA model for DDP mode")
    elif config.train.use_fsdp and config.train.ema_update:
        logger.warning("EMA update is not supported in FSDP mode. Setting model_ema to None.")
        model_ema = None
    else:
        model_ema = None

    logger.info(
        colored(
            f"{model.__class__.__name__}:{config.model.model}, "
            f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M",
            "green",
            attrs=["bold"],
        )
    )

    if config.train.use_fsdp:
        model_instance = deepcopy(model)
    elif model_ema is not None:
        model_instance = deepcopy(model_ema)
    else:
        model_instance = model

    # 4-1. load model
    if args.load_from is not None:
        config.model.load_from = args.load_from
    if config.model.load_from is not None and load_from:
        # Custom checkpoint loading for image-to-image training
        # Load checkpoint and remove y_embedder keys that are incompatible
        from tools.download import find_model
        
        logger.info("Loading checkpoint for image-to-image training...")
        checkpoint = find_model(config.model.load_from)
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        # Remove y_embedder keys that cause size mismatch
        keys_to_remove = [key for key in state_dict.keys() if key.startswith("y_embedder")]
        for key in keys_to_remove:
            del state_dict[key]
            logger.info(f"Removed incompatible key: {key}")
        
        # Load state dict with removed keys
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        logger.info(f"Missing keys after loading: {missing}")
        logger.info(f"Unexpected keys after loading: {unexpected}")

    # 4-2. model growth
    if config.model_growth is not None:
        assert config.model.load_from is None
        model_growth_initializer = ModelGrowthInitializer(model, config.model_growth)
        model = model_growth_initializer.initialize(
            strategy=config.model_growth.init_strategy, **config.model_growth.init_params
        )

    if config.train.ema_update and not config.train.use_fsdp and model_ema is not None:
        ema_update(model_ema, model, 0.0)

    # 5. build dataloader for image pairs
    config.data.data_dir = config.data.data_dir if isinstance(config.data.data_dir, list) else [config.data.data_dir]
    config.data.data_dir = [
        data if data.startswith(("https://", "http://", "gs://", "/", "~")) else osp.abspath(osp.expanduser(data))
        for data in config.data.data_dir
    ]
    num_replicas = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    
    # Override dataset type for image-to-image
    config.data.type = "SanaImg2ImgDataset"
    
    dataset = build_dataset(
        asdict(config.data),
        resolution=image_size,
        aspect_ratio_type=config.model.aspect_ratio_type,
        real_prompt_ratio=0.0,  # No text prompts
        max_length=0,  # No text needed
        config=config,
        caption_proportion=0.0,  # No captions
        sort_dataset=config.data.sort_dataset,
        vae_downsample_rate=config.vae.vae_downsample_rate,
    )
    accelerator.wait_for_everyone()
    if config.model.multi_scale:
        drop_last = True
        uuid = hashlib.sha256("-".join(config.data.data_dir).encode()).hexdigest()[:8]
        cache_dir = osp.expanduser(f"~/.cache/_wids_batchsampler_cache")
        os.makedirs(cache_dir, exist_ok=True)
        base_pattern = (
            f"{cache_dir}/{getpass.getuser()}-{uuid}-sort_dataset{config.data.sort_dataset}"
            f"-hq_only{config.data.hq_only}-valid_num{config.data.valid_num}"
            f"-aspect_ratio{len(dataset.aspect_ratio)}-droplast{drop_last}"
            f"dataset_len{len(dataset)}"
        )
        cache_file = f"{base_pattern}-num_replicas{num_replicas}-rank{rank}"
        for i in config.data.data_dir:
            cache_file += f"-{i}"
        cache_file += ".json"

        sampler = DistributedRangedSampler(dataset, num_replicas=num_replicas, rank=rank)
        batch_sampler = AspectRatioBatchSampler(
            sampler=sampler,
            dataset=dataset,
            batch_size=config.train.train_batch_size,
            aspect_ratios=dataset.aspect_ratio,
            drop_last=drop_last,
            ratio_nums=dataset.ratio_nums,
            config=config,
            valid_num=config.data.valid_num,
            hq_only=config.data.hq_only,
            cache_file=cache_file,
            caching=args.caching,
            clipscore_filter_thres=args.data.del_img_clip_thr,
        )
        train_dataloader = build_dataloader(dataset, batch_sampler=batch_sampler, num_workers=config.train.num_workers)
        train_dataloader_len = len(train_dataloader)
        logger.info(f"rank-{rank} Cached file len: {len(train_dataloader.batch_sampler.cached_idx)}")
    else:
        sampler = DistributedRangedSampler(dataset, num_replicas=num_replicas, rank=rank)
        train_dataloader = build_dataloader(
            dataset,
            num_workers=config.train.num_workers,
            batch_size=config.train.train_batch_size,
            shuffle=False,
            sampler=sampler,
        )
        train_dataloader_len = len(train_dataloader)
    load_vae_feat = getattr(train_dataloader.dataset, "load_vae_feat", False)

    # 6. build optimizer and lr scheduler
    lr_scale_ratio = 1
    if getattr(config.train, "auto_lr", None):
        lr_scale_ratio = auto_scale_lr(
            config.train.train_batch_size * get_world_size() * config.train.gradient_accumulation_steps,
            config.train.optimizer,
            **config.train.auto_lr,
        )
    optimizer = build_optimizer(model, config.train.optimizer)
    if config.train.lr_schedule_args and config.train.lr_schedule_args.get("num_warmup_steps", None):
        config.train.lr_schedule_args["num_warmup_steps"] = (
            config.train.lr_schedule_args["num_warmup_steps"] * num_replicas
        )
    lr_scheduler = build_lr_scheduler(config.train, optimizer, train_dataloader, lr_scale_ratio)
    logger.warning(
        f"{colored(f'Basic Setting: ', 'green', attrs=['bold'])}"
        f"lr: {config.train.optimizer['lr']:.5f}, bs: {config.train.train_batch_size}, gc: {config.train.grad_checkpointing}, "
        f"gc_accum_step: {config.train.gradient_accumulation_steps}, qk norm: {config.model.qk_norm}, "
        f"fp32 attn: {config.model.fp32_attention}, attn type: {config.model.attn_type}, ffn type: {config.model.ffn_type}, "
        f"precision: {config.model.mixed_precision}, img2img: True"
    )

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")

    start_epoch = 0
    start_step = 0
    total_steps = train_dataloader_len * config.train.num_epochs

    # 7. Resume training
    if config.model.resume_from is not None and config.model.resume_from["checkpoint"] is not None:
        rng_state = None
        ckpt_path = osp.join(config.work_dir, "checkpoints")
        check_flag = osp.exists(ckpt_path) and len(os.listdir(ckpt_path)) != 0

        if config.model.resume_from["checkpoint"] == "latest":
            if check_flag:
                config.model.resume_from["resume_optimizer"] = True
                config.model.resume_from["resume_lr_scheduler"] = True
                checkpoints = os.listdir(ckpt_path)
                if "latest.pth" in checkpoints and osp.exists(osp.join(ckpt_path, "latest.pth")):
                    config.model.resume_from["checkpoint"] = osp.realpath(osp.join(ckpt_path, "latest.pth"))
                else:
                    checkpoints = [i for i in checkpoints if i.startswith("epoch_")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.replace(".pth", "").split("_")[3]))
                    config.model.resume_from["checkpoint"] = osp.join(ckpt_path, checkpoints[-1])
            else:
                config.model.resume_from["resume_optimizer"] = config.train.load_from_optimizer
                config.model.resume_from["resume_lr_scheduler"] = config.train.load_from_lr_scheduler
                config.model.resume_from["checkpoint"] = config.model.load_from

        if config.model.resume_from["checkpoint"] is not None:
            _, missing, unexpected, _ = load_checkpoint(
                **config.model.resume_from,
                model=model,
                model_ema=model_ema if not config.train.use_fsdp else None,
                FSDP=config.train.use_fsdp,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                null_embed_path=null_embed_path,
            )

            logger.warning(f"Missing keys: {missing}")
            logger.warning(f"Unexpected keys: {unexpected}")

            path = osp.basename(config.model.resume_from["checkpoint"])
        try:
            start_epoch = int(path.replace(".pth", "").split("_")[1]) - 1
            start_step = int(path.replace(".pth", "").split("_")[3])
        except:
            pass

    # 8. Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model = accelerator.prepare(model)
    if model_ema is not None and not config.train.use_fsdp:
        model_ema = accelerator.prepare(model_ema)
    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    # load everything except model when resume
    if (
        config.train.use_fsdp
        and config.model.resume_from is not None
        and config.model.resume_from["checkpoint"] is not None
        and config.model.resume_from["resume_optimizer"]
        and config.model.resume_from["resume_lr_scheduler"]
    ):
        logger.info(f"FSDP resume: Loading optimizer, scheduler, scaler, random_states...")
        accelerator.load_state(
            os.path.join(config.model.resume_from["checkpoint"], "model"),
            state_dict_key=["optimizer", "scheduler", "scaler", "random_states"],
        )

    set_random_seed((start_step + 1) // config.train.save_model_steps + int(os.environ["LOCAL_RANK"]))
    logger.info(f'Set seed: {(start_step + 1) // config.train.save_model_steps + int(os.environ["LOCAL_RANK"])}')

    # Start Training
    train(
        config=config,
        args=args,
        accelerator=accelerator,
        model=model,
        model_ema=model_ema,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        train_diffusion=train_diffusion,
        logger=logger,
    )


if __name__ == "__main__":
    main() 