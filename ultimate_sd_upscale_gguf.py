import math
import torch
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
                "tile_width": ("INT", { "default": 512, "min": 256, "max": 2048, "step": 64 }),
                "tile_height": ("INT", { "default": 512, "min": 256, "max": 2048, "step": 64 }),
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
        self, image, noise, guider, sampler, sigmas, vae, upscale_by, tile_width, tile_height, 
        mask_blur, transition_sharpness, tile_padding, seam_fix_mode, seam_fix_width, seam_fix_mask_blur, seam_fix_padding, force_uniform_tiles
    ):
        """Upscale an image using tiled diffusion with seam fixing
        
        Args:
            image (torch.Tensor): Input image to upscale
            noise (torch.Tensor): Noise tensor
            guider: Guidance model
            sampler: Sampler instance
            sigmas: Sampling sigmas
            vae: VAE model
            upscale_by (float): Factor to upscale by
            tile_width (int): Width of tiles
            tile_height (int): Height of tiles
            mask_blur (float): Blur for tile masks
            transition_sharpness (float): Sharpness of transitions
            tile_padding (int): Padding around tiles
            seam_fix_mode (str): Mode for seam fixing
            seam_fix_width (int): Width for seam fixing
            seam_fix_mask_blur (float): Blur for seam masks
            seam_fix_padding (int): Padding for seam fixing
            force_uniform_tiles (bool): Force tiles to be uniform size
            
        Returns:
            torch.Tensor: Upscaled image
        """
        device = image.device
        
        # Create upscale settings
        settings = UpscaleSettings(
            width=image.shape[3],
            height=image.shape[2],
            upscale_by=upscale_by,
            tile_width=tile_width,
            tile_height=tile_height,
            force_uniform_tiles=force_uniform_tiles
        )
        
        # Initialize result tensor
        result = torch.zeros(
            (image.shape[0], image.shape[1], settings.target_height, settings.target_width),
            device=device
        )
        
        # Process each tile
        for tile_y in range(settings.num_tiles_y):
            for tile_x in range(settings.num_tiles_x):
                # Get tile coordinates
                x1, x2, y1, y2 = settings.get_tile_coordinates(tile_x, tile_y)
                
                # Calculate padded coordinates
                pad_x1 = max(0, x1 - tile_padding)
                pad_x2 = min(settings.target_width, x2 + tile_padding)
                pad_y1 = max(0, y1 - tile_padding)
                pad_y2 = min(settings.target_height, y2 + tile_padding)
                
                # Create mask for this tile
                mask = torch.zeros(
                    (1, 1, pad_y2 - pad_y1, pad_x2 - pad_x1),
                    device=device
                )
                
                # Set core tile area to 1
                core_x1 = x1 - pad_x1
                core_x2 = x2 - pad_x1
                core_y1 = y1 - pad_y1
                core_y2 = y2 - pad_y1
                mask[:, :, core_y1:core_y2, core_x1:core_x2] = 1
                
                # Blur the mask if specified
                if mask_blur > 0:
                    adjusted_blur = mask_blur * transition_sharpness
                    kernel_size = int(adjusted_blur * 2 + 1)
                    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device)
                    kernel = kernel / kernel.numel()
                    mask = torch.nn.functional.conv2d(
                        mask,
                        kernel,
                        padding=mask_blur
                    )
                    mask = torch.clamp(mask, 0, 1)
                
                # Extract padded region from image
                tile_image = image[:, :, 
                    pad_y1 // upscale_by:pad_y2 // upscale_by,
                    pad_x1 // upscale_by:pad_x2 // upscale_by
                ]
                
                # Process tile
                latent = Sampler.encode(tile_image, vae)
                sampled = sampler.sample(noise, latent, guider, sigmas)
                tile_result = vae.decode(sampled["samples"])
                
                # Blend tile into result using mask
                result[:, :, pad_y1:pad_y2, pad_x1:pad_x2] = \
                    tile_result * mask + \
                    result[:, :, pad_y1:pad_y2, pad_x1:pad_x2] * (1 - mask)
        
        # Fix seams if requested
        if seam_fix_mode != "None":
            seam_fixer = SeamFixer(
                seam_fix_mode,
                seam_fix_width,
                seam_fix_mask_blur,
                seam_fix_padding,
                transition_sharpness,
                settings,
                device
            )
            result = seam_fixer.fix_seams(result, vae, sampler, noise, guider, sigmas)
        
        return (result,)  # Return as tuple for ComfyUI