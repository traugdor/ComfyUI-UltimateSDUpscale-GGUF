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
from .upscale_settings import UpscaleSettings
from .sampler import Sampler
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
                "upscale_by": ("FLOAT", { "default": 2.0, "min": 1.0, "max": 8.0, "step": 0.1 }),
                "max_tile_size": ("INT", { "default": 512, "min": 256, "max": 2048, "step": 64 }),
                "mask_blur": ("INT", { "default": 8, "min": 0, "max": 64, "step": 1 }),
                "transition_sharpness": ("FLOAT", { "default": 0.33, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "tile_padding": ("INT", { "default": 32, "min": 0, "max": 128, "step": 8 }),
                "seam_fix_mode": ("STRING", { "default": "None", "options": ["None", "Band Pass", "Half Tile", "Half Tile + Intersections"] }),
                "seam_fix_width": ("INT", { "default": 64, "min": 0, "max": 8192, "step": 8 }),
                "seam_fix_mask_blur": ("INT", { "default": 8, "min": 0, "max": 64, "step": 1 }),
                "seam_fix_padding": ("INT", { "default": 16, "min": 0, "max": 128, "step": 8 }),
                "force_uniform_tiles": ("BOOLEAN", { "default": True })
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(
            self, image, noise, guider, sampler, sigmas, vae, upscale_by, max_tile_size,
            mask_blur, transition_sharpness, tile_padding, seam_fix_mode, seam_fix_width, 
            seam_fix_mask_blur, seam_fix_padding, force_uniform_tiles
        ):
        
        # Step 1: Initialize settings
        settings = UpscaleSettings(
            target_width=int(image.shape[2] * upscale_by),
            target_height=int(image.shape[1] * upscale_by),
            max_tile_size=max_tile_size,
            tile_padding=tile_padding,
            force_uniform_tiles=force_uniform_tiles
        )

        # Step 2: Upscale image using bicubic
        samples = image.movedim(-1,1)  # BHWC to BCHW
        image = comfy.utils.common_upscale(samples, settings.sampling_width, settings.sampling_height, "bicubic", "disabled")
        image = image.movedim(1,-1)  # BCHW to BHWC
        
        # Create output tensor on CPU
        output = torch.zeros_like(image)
        
        # Process each tile
        for tile_y in range(settings.num_tiles_y):
            for tile_x in range(settings.num_tiles_x):
                
                # Get tile coordinates with padding
                x1, x2, y1, y2, pad_x1, pad_x2, pad_y1, pad_y2 = settings.get_tile_coordinates(tile_x, tile_y, tile_padding)
                
                # Step 3: Extract padded tile
                tile = image[:, pad_y1:pad_y2, pad_x1:pad_x2, :].clone()  # Clone to ensure memory separation
                
                # Step 4-5: Create and pad mask
                mask = torch.ones((1, 1, y2-y1, x2-x1))
                if tile_padding > 0:
                    padded_mask = torch.nn.functional.pad(mask, (tile_padding,)*4, mode='constant', value=0)
                else:
                    padded_mask = mask

                # Step 6: Blur mask edges (on CPU)
                if mask_blur > 0:
                    kernel_size = mask_blur * 2 + 1
                    kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
                    padded_mask = torch.nn.functional.conv2d(
                        padded_mask, kernel, padding=mask_blur
                    )
                    padded_mask = torch.clamp(padded_mask * (1 + transition_sharpness), 0, 1)

                # Step 7: Crop mask edges for edge tiles
                if pad_x1 == 0:  # Left edge
                    padded_mask = padded_mask[:, :, :, tile_padding:]
                if pad_x2 == settings.target_width:  # Right edge
                    padded_mask = padded_mask[:, :, :, :-tile_padding]
                if pad_y1 == 0:  # Top edge
                    padded_mask = padded_mask[:, :, tile_padding:, :]
                if pad_y2 == settings.target_height:  # Bottom edge
                    padded_mask = padded_mask[:, :, :-tile_padding, :]

                # Step 8-9: Sample and decode tile
                tile_latent = Sampler.encode(tile, vae)
                
                #scale mask to match tile size
                latent_h, latent_w = tile_latent["samples"].shape[2:]
                padded_mask = F.interpolate(padded_mask, size=(latent_h, latent_w), mode='bilinear')
                
                tile_samples = {"samples": tile_latent["samples"], "noise_mask": padded_mask}
                
                tile_samples = Sampler.sample(noise, guider, sampler, sigmas, tile_samples)
                tile = vae.decode(tile_samples["samples"])

                # Step 10: Paste tile back with mask
                output[:, y1:y2, x1:x2, :] = tile * mask + \
                    tile[:, :y2-y1, :x2-x1, :] * (1 - mask)

        # Scale image back down to target size
        samples = output.movedim(-1,1)  # BHWC to BCHW
        image = comfy.utils.common_upscale(samples, settings.target_width, settings.target_height, "area", "disabled")
        output = image.movedim(1,-1)  # BCHW to BHWC
        
        return (output,)