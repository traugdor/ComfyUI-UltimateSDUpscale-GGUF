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
        
        # Create output tensor on CPU to save VRAM
        output = torch.zeros_like(image, device='cpu')
        
        # Initialize storage for our phases
        latents = []
        tile_positions = []
        
        # Phase 1: VAE Encode
        print("Phase 1: VAE Encoding")
        print("Cleaning VRAM...")
        SamplerHelper.force_memory_cleanup(True)  # Ensure clean VRAM state
        
        for tile_y in range(settings.num_tiles_y):
            for tile_x in range(settings.num_tiles_x):
                # Get tile coordinates with padding
                x1, x2, y1, y2, pad_x1, pad_x2, pad_y1, pad_y2 = settings.get_tile_coordinates(tile_x, tile_y, tile_padding)
                
                # Extract padded tile
                tile = image[:, pad_y1:pad_y2, pad_x1:pad_x2, :].clone()
                
                # Encode tile
                latent = Sampler.encode(tile, vae)
                latents.append(latent)
                tile_positions.append((x1, x2, y1, y2, pad_x1, pad_x2, pad_y1, pad_y2))
                
                del tile
                torch.cuda.empty_cache()
        
        # Unload VAE to free memory
        print("Unloading VAE...")
        SamplerHelper.force_memory_cleanup(True)
        
        # Phase 2: Process Latents
        print("Phase 2: Processing Latents")
        processed_latents = SamplerHelper.process_latent_batch(latents, noise, guider, sampler, sigmas)
        
        # Clear original latents
        del latents
        print("Cleaning VRAM...")
        SamplerHelper.force_memory_cleanup(True)
        
        # Phase 3: VAE Decode
        print("Phase 3: VAE Decoding")
        decoded_tiles = []
        
        for processed_latent in processed_latents:
            # Decode latent
            tile = vae.decode(processed_latent["samples"])
            decoded_tiles.append(tile.cpu())  # Move to CPU immediately
            
            del processed_latent
            torch.cuda.empty_cache()
        
        del processed_latents
        print("Unloading VAE...")
        SamplerHelper.force_memory_cleanup(True)
        
        # Phase 4: Stitch Tiles
        print("Phase 4: Stitching Tiles")
        # Create gaussian blend kernel
        blend_kernel = SamplerHelper.create_gaussian_blend_kernel(tile_padding, image.device)
        
        for tile, pos in zip(decoded_tiles, tile_positions):
            x1, x2, y1, y2, _, _, _, _ = pos
            # Move tile and relevant output section to GPU for blending
            tile_gpu = tile.to(image.device)
            output_section = output[:, y1:y2, x1:x2, :].to(image.device)
            
            # Blend and store
            blended = SamplerHelper.blend_tile_edges(tile_gpu, output_section, blend_kernel)
            output[:, y1:y2, x1:x2, :] = blended.cpu()
            
            del tile_gpu, output_section
            torch.cuda.empty_cache()
        
        # Clean up
        del decoded_tiles, tile_positions
        SamplerHelper.force_memory_cleanup(True)
        
        # Scale image back down to target size
        output = output.to(image.device)  # Move back to GPU for scaling
        samples = output.movedim(-1,1)  # BHWC to BCHW
        output = comfy.utils.common_upscale(samples, settings.target_width, settings.target_height, "area", "disabled")
        output = output.movedim(1,-1)  # BCHW to BHWC
        
        return (output,)