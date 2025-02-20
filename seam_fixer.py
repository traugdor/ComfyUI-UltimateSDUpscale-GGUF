import torch
import os
import sys

# Add ComfyUI path to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMFY_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if COMFY_DIR not in sys.path:
    sys.path.append(COMFY_DIR)

from .upscale_settings import UpscaleSettings
from .sampler import Sampler

class SeamFixer:
    VALID_MODES = ["None", "Band Pass", "Half Tile", "Half Tile + Intersections"]
    
    def __init__(self, mode, width, mask_blur, padding, transition_sharpness, settings, device):
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid seam fix mode: {mode}. Must be one of {self.VALID_MODES}")
            
        self.mode = mode
        self.width = width
        self.mask_blur = mask_blur
        self.padding = padding
        self.upscale_settings = settings
        self.transition_sharpness = transition_sharpness
        self.device = device
    
    def get_band_coordinates(self):
        vertical_bands = []
        horizontal_bands = []
        
        # Vertical bands (along tile columns)
        for x in range(1, self.upscale_settings.num_tiles_x):
            # Calculate x position where tiles meet
            seam_x = x * self.upscale_settings.tile_width
            start_x = max(0, seam_x - self.width)
            end_x = min(self.upscale_settings.target_width, seam_x + self.width)
            # Band goes full height
            vertical_bands.append((start_x, end_x, 0, self.upscale_settings.target_height))
        
        # Horizontal bands (along tile rows)
        for y in range(1, self.upscale_settings.num_tiles_y):
            # Calculate y position where tiles meet
            seam_y = y * self.upscale_settings.tile_height
            start_y = max(0, seam_y - self.width)
            end_y = min(self.upscale_settings.target_height, seam_y + self.width)
            # Band goes full width
            horizontal_bands.append((0, self.upscale_settings.target_width, start_y, end_y))
        
        return vertical_bands, horizontal_bands
    
    def get_half_tile_coordinates(self):
        vertical_halves = []
        horizontal_halves = []
        
        # Vertical seams (process right half of left tile and left half of right tile)
        for x in range(1, self.upscale_settings.num_tiles_x):
            seam_x = x * self.upscale_settings.tile_width
            
            # Right half of left tile
            left_half = (
                seam_x - self.upscale_settings.tile_width//2,  # start at middle of left tile
                seam_x + self.padding,  # extend slightly into right tile
                0,  # full height
                self.upscale_settings.target_height
            )
            
            # Left half of right tile
            right_half = (
                seam_x - self.padding,  # start slightly in left tile
                seam_x + self.upscale_settings.tile_width//2,  # end at middle of right tile
                0,  # full height
                self.upscale_settings.target_height
            )
            
            vertical_halves.extend([left_half, right_half])
        
        # Horizontal seams (process bottom half of top tile and top half of bottom tile)
        for y in range(1, self.upscale_settings.num_tiles_y):
            seam_y = y * self.upscale_settings.tile_height
            
            # Bottom half of top tile
            top_half = (
                0,  # full width
                self.upscale_settings.target_width,
                seam_y - self.upscale_settings.tile_height//2,  # start at middle of top tile
                seam_y + self.padding  # extend slightly into bottom tile
            )
            
            # Top half of bottom tile
            bottom_half = (
                0,  # full width
                self.upscale_settings.target_width,
                seam_y - self.padding,  # start slightly in top tile
                seam_y + self.upscale_settings.tile_height//2  # end at middle of bottom tile
            )
            
            horizontal_halves.extend([top_half, bottom_half])
        
        return vertical_halves, horizontal_halves
        
    def get_intersection_coordinates(self):
        intersections = []
        
        # For each internal tile corner (where 4 tiles meet)
        for y in range(1, self.upscale_settings.num_tiles_y):
            for x in range(1, self.upscale_settings.num_tiles_x):
                seam_x = x * self.upscale_settings.tile_width
                seam_y = y * self.upscale_settings.tile_height
                
                # Calculate the intersection region centered on the seam intersection
                # This creates a square region that overlaps with the half-tiles
                half_width = self.upscale_settings.tile_width // 4  # Quarter tile width
                half_height = self.upscale_settings.tile_height // 4  # Quarter tile height
                
                intersection = (
                    seam_x - half_width,  # start quarter tile left of seam
                    seam_x + half_width,  # end quarter tile right of seam
                    seam_y - half_height,  # start quarter tile above seam
                    seam_y + half_height   # end quarter tile below seam
                )
                
                intersections.append(intersection)
        
        return intersections
        
    def process_band(self, upscaled_image, band, vae, sampler, noise, guider, sigmas):
        start_x, end_x, start_y, end_y = band
        
        # Extract band region
        band_image = upscaled_image[:, start_y:end_y, start_x:end_x, :]
        
        # Create mask for blending (in BCHW for conv2d)
        mask = torch.zeros((1, 1, end_y - start_y, end_x - start_x), device=self.device)
        mask[:, :, :, :] = 1
        
        # Apply mask blur if specified
        if self.mask_blur > 0:
            adjusted_blur = self.mask_blur * self.transition_sharpness
            # Ensure kernel size is odd and not larger than input
            kernel_size = min(
                int(adjusted_blur * 2 + 1),
                min(end_y - start_y, end_x - start_x) - 1  # Leave at least 1 pixel
            )
            if kernel_size % 2 == 0:  # Make odd
                kernel_size -= 1
            if kernel_size > 0:  # Only apply if we have a valid kernel size
                kernel = torch.ones(1, 1, kernel_size, kernel_size, device=self.device)
                kernel = kernel / kernel.numel()
                mask = torch.nn.functional.conv2d(
                    mask,
                    kernel,
                    padding=kernel_size//2
                )
                mask = torch.clamp(mask, 0, 1)
        
        # Process through VAE and sampling (VAE expects BHWC format)
        latent = Sampler.encode(band_image, vae)
        latent["noise_mask"] = mask  # Noise mask stays in BCHW format
        
        sampled = Sampler.sample(noise, guider, sampler, sigmas, latent)
        processed_band = vae.decode(sampled["samples"])
        
        # Convert mask to BHWC for blending
        mask = mask.permute(0, 2, 3, 1)
        
        return processed_band, mask
        
    def fix_seams(self, upscaled_image, vae, sampler, noise, guider, sigmas):
        if self.mode == "None":
            return upscaled_image
            
        result_image = upscaled_image.clone()
            
        if self.mode == "Band Pass":
            vertical_bands, horizontal_bands = self.get_band_coordinates()
            
            # Process vertical bands
            for band in vertical_bands:
                processed_band, mask = self.process_band(
                    upscaled_image, band, vae, sampler, noise, guider, sigmas
                )
                start_x, end_x, start_y, end_y = band
                
                # Blend band back into image
                for c in range(upscaled_image.shape[-1]):
                    result_image[:, start_y:end_y, start_x:end_x, c] = \
                        processed_band[:, :, :, c] * mask[:, :, :, 0] + \
                        result_image[:, start_y:end_y, start_x:end_x, c] * (1 - mask[:, :, :, 0])
            
            # Process horizontal bands
            for band in horizontal_bands:
                processed_band, mask = self.process_band(
                    result_image, band, vae, sampler, noise, guider, sigmas
                )
                start_x, end_x, start_y, end_y = band
                
                # Blend band back into image
                for c in range(upscaled_image.shape[-1]):
                    result_image[:, start_y:end_y, start_x:end_x, c] = \
                        processed_band[:, :, :, c] * mask[:, :, :, 0] + \
                        result_image[:, start_y:end_y, start_x:end_x, c] * (1 - mask[:, :, :, 0])
                        
        elif self.mode in ["Half Tile", "Half Tile + Intersections"]:
            vertical_halves, horizontal_halves = self.get_half_tile_coordinates()
            
            # Process vertical half-tiles
            for half_tile in vertical_halves:
                processed_half, mask = self.process_band(
                    upscaled_image, half_tile, vae, sampler, noise, guider, sigmas
                )
                start_x, end_x, start_y, end_y = half_tile
                
                # Blend half-tile back into image
                for c in range(upscaled_image.shape[-1]):
                    result_image[:, start_y:end_y, start_x:end_x, c] = \
                        processed_half[:, :, :, c] * mask[:, :, :, 0] + \
                        result_image[:, start_y:end_y, start_x:end_x, c] * (1 - mask[:, :, :, 0])
            
            # Process horizontal half-tiles
            for half_tile in horizontal_halves:
                processed_half, mask = self.process_band(
                    result_image, half_tile, vae, sampler, noise, guider, sigmas
                )
                start_x, end_x, start_y, end_y = half_tile
                
                # Blend half-tile back into image
                for c in range(upscaled_image.shape[-1]):
                    result_image[:, start_y:end_y, start_x:end_x, c] = \
                        processed_half[:, :, :, c] * mask[:, :, :, 0] + \
                        result_image[:, start_y:end_y, start_x:end_x, c] * (1 - mask[:, :, :, 0])
            
            # Process intersections if in intersection mode
            if self.mode == "Half Tile + Intersections":
                intersections = self.get_intersection_coordinates()
                
                # Process each intersection region
                for intersection in intersections:
                    processed_intersection, mask = self.process_band(
                        result_image, intersection, vae, sampler, noise, guider, sigmas
                    )
                    start_x, end_x, start_y, end_y = intersection
                    
                    # Use radial gradient for intersection mask
                    # This creates a circular blend that smoothly transitions in all directions
                    center_x = (end_x - start_x) // 2
                    center_y = (end_y - start_y) // 2
                    y, x = torch.meshgrid(
                        torch.arange(end_y - start_y, device=self.device),
                        torch.arange(end_x - start_x, device=self.device),
                        indexing='ij'
                    )
                    radius = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
                    max_radius = min(center_x, center_y)
                    radial_mask = torch.clamp(1 - radius / max_radius, 0, 1)
                    
                    # Blend intersection back into image
                    radial_mask = radial_mask.unsqueeze(-1)  # Add channel dimension
                    for c in range(upscaled_image.shape[-1]):
                        result_image[:, start_y:end_y, start_x:end_x, c] = \
                            processed_intersection[:, :, :, c] * radial_mask[:, :, :, 0] + \
                            result_image[:, start_y:end_y, start_x:end_x, c] * (1 - radial_mask[:, :, :, 0])
            
        return result_image
