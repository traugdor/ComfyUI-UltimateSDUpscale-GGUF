import math
import torch
import os
import sys
import time
import logging
import torch.cuda
import torch.nn.functional as F

# Add ComfyUI path to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMFY_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if COMFY_DIR not in sys.path:
    sys.path.append(COMFY_DIR)

import comfy.utils
import comfy_extras.nodes_upscale_model as numodel
from .upscale_settings import UpscaleSettings
from .sampler import SamplerHelper, Sampler
from .seam_fixer import SeamFixer

class UltimateSDUpscaleGGUF:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                "image": ("IMAGE", ),
                "noise": ("NOISE", ),
                "guider": ("GUIDER", ),
                "sampler": ("SAMPLER", ),
                "sigmas": ("SIGMAS", ),
                "vae": ("VAE", ),
                "upscale_model": ("UPSCALE_MODEL",),
                "upscale_by": ("FLOAT", { "default": 2.0, "min": 1.0, "max": 8.0, "step": 0.1 }),
                "max_tile_size": ("INT", { "default": 512, "min": 256, "max": 2048, "step": 64 }),
                "mask_blur": ("INT", { "default": 8, "min": 0, "max": 64, "step": 1 }),
                "transition_sharpness": ("FLOAT", { "default": 0.333, "min": 0.125, "max": 1.0, "step": 0.001 }),
                "tile_padding": ("INT", { "default": 32, "min": 0, "max": 128, "step": 8 }),
                "seam_fix_mode": ("STRING", { "default": "None", "options": ["None", "Band Pass", "Half Tile", "Half Tile + Intersections"] }),
                "seam_fix_width": ("INT", { "default": 64, "min": 0, "max": 8192, "step": 8 }),
                "seam_fix_mask_blur": ("INT", { "default": 8, "min": 0, "max": 64, "step": 1 }),
                "seam_fix_padding": ("INT", { "default": 16, "min": 0, "max": 128, "step": 8 }),
                "force_uniform_tiles": ("BOOLEAN", { "default": True })
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")  # (upscaled_image, tiles, masks)
    RETURN_NAMES = ("upscaled", "tiles", "masks")
    OUTPUT_IS_LIST = (False, True, True)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(
            self, image, noise, guider, sampler, sigmas, vae, upscale_model, upscale_by, max_tile_size,
            mask_blur, transition_sharpness, tile_padding, seam_fix_mode, seam_fix_width, 
            seam_fix_mask_blur, seam_fix_padding, force_uniform_tiles
        ):
        settings = UpscaleSettings(
            target_width=int(image.shape[2] * upscale_by),
            target_height=int(image.shape[1] * upscale_by),
            max_tile_size=max_tile_size,
            tile_padding=tile_padding,
            force_uniform_tiles=force_uniform_tiles
        )

        upScalerWithModel = numodel.ImageUpscaleWithModel()
        image_tuple = upScalerWithModel.upscale(upscale_model, image)
        image = image_tuple[0]
        samples = image.movedim(-1,1)
        image = comfy.utils.common_upscale(samples, settings.sampling_width, settings.sampling_height, "area", "disabled")
        image = image.movedim(1,-1)
        
        output = image.to('cpu')
        latents = []
        tile_positions = []
        tile_masks = []
        all_tiles = []
        all_masks = []
        
        SamplerHelper.force_memory_cleanup(True)

        for tile_y in range(settings.num_tiles_y):
            for tile_x in range(settings.num_tiles_x):
                x1, x2, y1, y2, pad_x1, pad_x2, pad_y1, pad_y2 = settings.get_tile_coordinates(tile_x, tile_y, tile_padding)
                
                if tile_padding > 0:
                    pad = tile_padding
                    full_h, full_w = (y2-y1) + pad*2, (x2-x1) + pad*2
                    mask = torch.zeros((1, 1, full_h, full_w), device=image.device)
                    y_coords = torch.arange(full_h, device=image.device).view(-1, 1)
                    x_coords = torch.arange(full_w, device=image.device).view(1, -1)
                    tile_y1, tile_y2 = pad, pad + (y2-y1)
                    tile_x1, tile_x2 = pad, pad + (x2-x1)
                    dist_from_y1 = torch.abs(y_coords - tile_y1)
                    dist_from_y2 = torch.abs(y_coords - tile_y2)
                    dist_from_x1 = torch.abs(x_coords - tile_x1)
                    dist_from_x2 = torch.abs(x_coords - tile_x2)
                    y_dist = torch.minimum(dist_from_y1, dist_from_y2)
                    x_dist = torch.minimum(dist_from_x1, dist_from_x2)
                    y_dist = torch.where((y_coords >= tile_y1) & (y_coords <= tile_y2), 0, y_dist)
                    x_dist = torch.where((x_coords >= tile_x1) & (x_coords <= tile_x2), 0, x_dist)
                    dist = torch.sqrt(y_dist**2 + x_dist**2)
                    falloff = 1.0 - torch.clamp(dist / pad, min=0, max=1)
                    mask[0, 0] = falloff
                    
                    if mask_blur > 0:
                        kernel_size = min(pad * 2 - 1, 63)
                        sigma = pad / 2 * math.ceil(1.0 / transition_sharpness) / 4
                        x = torch.arange(-(kernel_size//2), kernel_size//2 + 1, device=image.device).float()
                        gaussian = torch.exp(-(x**2)/(2*sigma**2))
                        gaussian = gaussian / gaussian.sum()
                        kernel = gaussian.view(1, 1, -1, 1) @ gaussian.view(1, 1, 1, -1)
                        mask = F.conv2d(mask, kernel, padding=(kernel_size-1) // 2)
                    
                    mask[:, :, pad:pad+(y2-y1), pad:pad+(x2-x1)] = 1.0
                    
                    x_start = 0 if tile_x > 0 else pad
                    x_end = full_w if tile_x < settings.num_tiles_x - 1 else full_w - pad
                    y_start = 0 if tile_y > 0 else pad
                    y_end = full_h if tile_y < settings.num_tiles_y - 1 else full_h - pad
                    
                    mask = mask[:, :, y_start:y_end, x_start:x_end]
                else:
                    mask = torch.ones((1, 1, y2-y1, x2-x1), device=image.device)
                
                mask = torch.clamp(mask, 0, 1)
                tile = image[:, pad_y1:pad_y2, pad_x1:pad_x2, :].clone()
                latent = Sampler.encode(tile, vae)
                latent_h, latent_w = latent["samples"].shape[2:4]
                mask_latent = F.interpolate(mask, size=(latent_h, latent_w), mode='bilinear')
                latents.append(latent)
                tile_positions.append((x1, x2, y1, y2, pad_x1, pad_x2, pad_y1, pad_y2))
                tile_masks.append(mask)
                del tile, mask, mask_latent
                torch.cuda.empty_cache()
        
        SamplerHelper.force_memory_cleanup(True)
        
        processed_latents = SamplerHelper.process_latent_batch(latents, noise, guider, sampler, sigmas)
        #processed_latents = latents
        del latents
        SamplerHelper.force_memory_cleanup(True)
        decoded_tiles = []
        for processed_latent in processed_latents:
            tile = vae.decode(processed_latent["samples"])
            decoded_tiles.append(tile.cpu())
            del processed_latent
            torch.cuda.empty_cache()
        
        del processed_latents
        SamplerHelper.force_memory_cleanup(True)
        
        for tile, pos, mask in zip(decoded_tiles, tile_positions, tile_masks):
            x1, x2, y1, y2, tpad_x1, tpad_x2, tpad_y1, tpad_y2 = pos
            output_slice = output[:, tpad_y1:tpad_y2, tpad_x1:tpad_x2, :]
            tile_gpu = tile.to(image.device)
            mask = mask.to(image.device)
            mask = mask.movedim(1, -1)
            mask = mask.expand(-1, -1, -1, 3)
            blended = tile_gpu * mask + output_slice.to(image.device) * (1 - mask)
            output[:, tpad_y1:tpad_y2, tpad_x1:tpad_x2, :] = blended
            all_tiles.append(tile_gpu.cpu())
            all_masks.append(mask.cpu())
            del tile_gpu, mask
            torch.cuda.empty_cache()
        
        del decoded_tiles, tile_positions, tile_masks
        SamplerHelper.force_memory_cleanup(True)
        samples = output.movedim(-1,1)
        output = comfy.utils.common_upscale(samples, settings.target_width, settings.target_height, "lanczos", "disabled")
        output = output.movedim(1,-1)
        
        return (output, all_tiles, all_masks)