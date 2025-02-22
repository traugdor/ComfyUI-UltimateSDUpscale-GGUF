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
        tile_masks = []
        
        # Phase 1: VAE Encode
        print("Phase 1: VAE Encoding and Mask Creation")
        print("Cleaning VRAM...")
        SamplerHelper.force_memory_cleanup(True)  # Ensure clean VRAM state

        for tile_y in range(settings.num_tiles_y):
            for tile_x in range(settings.num_tiles_x):
                # Get tile coordinates with padding
                x1, x2, y1, y2, pad_x1, pad_x2, pad_y1, pad_y2 = settings.get_tile_coordinates(tile_x, tile_y, tile_padding)
                
                # 1. Create base mask of tile size
                mask = torch.ones((1, 1, y2-y1, x2-x1), device=image.device)
                
                if tile_padding // 2 > 1:
                    # 2. Pad the entire mask using tile_padding
                    pad_size = tile_padding // 2
                    padded_mask = F.pad(mask, (pad_size, pad_size, pad_size, pad_size))
                    
                    # 3. Apply gaussian blur
                    kernel_size = int(mask_blur * 2 + 1)
                    sigma = mask_blur * transition_sharpness # use transition sharpness to adjust sigma
                    x = torch.arange(-(kernel_size//2), kernel_size//2 + 1, device=image.device).float()
                    gaussian = torch.exp(-(x**2)/(2*sigma**2))
                    gaussian = gaussian / gaussian.sum()
                    kernel = gaussian.view(1, 1, -1, 1) @ gaussian.view(1, 1, 1, -1)
                    
                    blurred = F.conv2d(padded_mask, kernel, padding=kernel_size//2)
                    
                    # 4. Crop mask back to padded tile size
                    mask = blurred[:, :, :pad_y2-pad_y1, :pad_x2-pad_x1]
                    mask = torch.clamp(mask, 0, 1)
                
                # Extract padded tile
                print(f"Padded Tile coordinates: {pad_x1} {pad_x2} {pad_y1} {pad_y2}")
                print(f"Extracting tile of size: {pad_x2-pad_x1}x{pad_y2-pad_y1}")
                print(f"     Original tile size: {x2-x1}x{y2-y1}")
                tile = image[:, pad_y1:pad_y2, pad_x1:pad_x2, :].clone()
                
                # Convert tile to latent and downscale mask
                latent = Sampler.encode(tile, vae)
                latent_h, latent_w = latent["samples"].shape[2:4]
                mask_latent = F.interpolate(mask, size=(latent_h, latent_w), mode='bilinear')
                # latent["noise_mask"] = mask_latent  # removing this for now. sample the entire tile.
                # we'll use the mask to blend it later.
                
                # Store for processing
                latents.append(latent)
                tile_positions.append((x1, x2, y1, y2, pad_x1, pad_x2, pad_y1, pad_y2))
                tile_masks.append(mask)  # Store original mask for final blending
                
                del tile, mask, mask_latent
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
        
        # Phase 4: Stitch Tiles with Blur Masks
        print("Phase 4: Stitching Tiles")
        
        for tile, pos, mask in zip(decoded_tiles, tile_positions, tile_masks):
            x1, x2, y1, y2, pad_x1, pad_x2, pad_y1, pad_y2 = pos
            
            # Move tile to GPU and expand mask to match channels
            tile_gpu = tile.to(image.device)
            mask = mask.to(image.device)
            
            # Convert mask from BCHW to BHWC format to match the image format
            mask = mask.movedim(1, -1)  # This moves the channel dimension to the end
            mask = mask.expand(-1, -1, -1, 3)  # Expand to 3 channels
            
            # Blend with existing content
            output[:, pad_y1:pad_y2, pad_x1:pad_x2, :] = (
                tile_gpu * mask + 
                output[:, pad_y1:pad_y2, pad_x1:pad_x2, :].to(image.device) * (1 - mask)
            )
            
            del tile_gpu, mask
            torch.cuda.empty_cache()
        
        # Clean up
        del decoded_tiles, tile_positions, tile_masks
        SamplerHelper.force_memory_cleanup(True)
        
        # Scale image back down to target size
        output = output.to(image.device)  # Move back to GPU for scaling
        samples = output.movedim(-1,1)  # BHWC to BCHW
        output = comfy.utils.common_upscale(samples, settings.target_width, settings.target_height, "bicubic", "disabled")
        output = output.movedim(1,-1)  # BCHW to BHWC
        
        return (output,)