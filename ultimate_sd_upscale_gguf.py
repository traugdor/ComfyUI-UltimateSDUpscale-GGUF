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
            self, image, noise, guider, sampler, sigmas, vae, upscale_model, upscale_by, max_tile_size,
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

        # Step 2: Upscale image using model (for better clarity and it just looks better in the output)
        upScalerWithModel = numodel.ImageUpscaleWithModel()
        image_tuple = upScalerWithModel.upscale(upscale_model, image)
        image = image_tuple[0]
        #downscale back to target size
        samples = image.movedim(-1,1)  # BHWC to BCHW
        image = comfy.utils.common_upscale(samples, settings.sampling_width, settings.sampling_height, "area", "disabled")
        image = image.movedim(1,-1)  # BHWC to BCHW
        
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
                
                if tile_padding > 0:
                    # Create a padded mask that's larger than we need
                    pad = tile_padding
                    full_h, full_w = (y2-y1) + pad*2, (x2-x1) + pad*2
                    mask = torch.zeros((1, 1, full_h, full_w), device=image.device)
                    
                    # Fill the center with ones (actual tile area)
                    mask[:, :, pad:pad+(y2-y1), pad:pad+(x2-x1)] = 1.0
                    
                    if mask_blur > 0:
                        # Apply gaussian blur
                        kernel_size = int(mask_blur * 2 + 1)
                        sigma = mask_blur * transition_sharpness
                        x = torch.arange(-(kernel_size//2), kernel_size//2 + 1, device=image.device).float()
                        gaussian = torch.exp(-(x**2)/(2*sigma**2))
                        gaussian = gaussian / gaussian.sum()
                        kernel = gaussian.view(1, 1, -1, 1) @ gaussian.view(1, 1, 1, -1)
                        
                        mask = F.conv2d(mask, kernel, padding=(kernel_size-1) // 2)
                    
                    # Fill the center with ones (actual tile area) again to ensure no edge artifacts
                    mask[:, :, pad:pad+(y2-y1), pad:pad+(x2-x1)] = 1.0
                    
                    # Crop the mask to the final padded tile size
                    # Only crop edges that touch image boundary
                    x_start = 0 if tile_x > 0 else pad
                    x_end = full_w if tile_x < settings.num_tiles_x - 1 else full_w - pad
                    y_start = 0 if tile_y > 0 else pad
                    y_end = full_h if tile_y < settings.num_tiles_y - 1 else full_h - pad
                    
                    mask = mask[:, :, y_start:y_end, x_start:x_end]
                else:
                    # If no padding, just use a mask of ones
                    mask = torch.ones((1, 1, y2-y1, x2-x1), device=image.device)
                
                mask = torch.clamp(mask, 0, 1)
                
                # Extract padded tile
                tile = image[:, pad_y1:pad_y2, pad_x1:pad_x2, :].clone()
                
                # Convert tile to latent and downscale mask
                latent = Sampler.encode(tile, vae)
                latent_h, latent_w = latent["samples"].shape[2:4]
                mask_latent = F.interpolate(mask, size=(latent_h, latent_w), mode='bilinear')
                
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
            
            # print(f"\nTile Position Info:")
            # print(f"Original bounds: ({x1}, {x2}, {y1}, {y2})")
            # print(f"Padded bounds: ({pad_x1}, {pad_x2}, {pad_y1}, {pad_y2})")
            # print(f"Tile shape: {tile.shape}")
            
            # Move tile to GPU and expand mask to match channels
            tile_gpu = tile.to(image.device)
            mask = mask.to(image.device)
            
            # print(f"Mask Info (before format change):")
            # print(f"Mask shape: {mask.shape}")
            # print(f"Mask value range: ({mask.min().item():.3f}, {mask.max().item():.3f})")
            # print(f"Mask edge values (left, right, top, bottom):")
            # print(f"  Left:   {mask[0,0,0,0].item():.3f}")
            # print(f"  Right:  {mask[0,0,-1,0].item():.3f}")
            # print(f"  Top:    {mask[0,0,0,0].item():.3f}")
            # print(f"  Bottom: {mask[0,0,0,-1].item():.3f}")
            
            # Convert mask from BCHW to BHWC format to match the image format
            mask = mask.movedim(1, -1)  # This moves the channel dimension to the end
            mask = mask.expand(-1, -1, -1, 3)  # Expand to 3 channels
            
            # print(f"Mask Info (after format change):")
            # print(f"Mask shape: {mask.shape}")
            # print(f"Expected shape based on padded bounds: ({pad_y2-pad_y1}, {pad_x2-pad_x1}, 3)")
            
            # Blend with existing content
            output_slice = output[:, pad_y1:pad_y2, pad_x1:pad_x2, :]
            print(f"Output slice shape: {output_slice.shape}")
            output[:, pad_y1:pad_y2, pad_x1:pad_x2, :] = (
                tile_gpu * mask + 
                output_slice.to(image.device) * (1 - mask)
            )
            
            del tile_gpu, mask
            torch.cuda.empty_cache()
        
        # Clean up
        del decoded_tiles, tile_positions, tile_masks
        SamplerHelper.force_memory_cleanup(True)
        
        # Scale image back down to target size
        output = output.to(image.device)  # Move back to GPU for scaling
        samples = output.movedim(-1,1)  # BHWC to BCHW
        output = comfy.utils.common_upscale(samples, settings.target_width, settings.target_height, "lanczos", "disabled")
        output = output.movedim(1,-1)  # BCHW to BHWC
        
        return (output,)