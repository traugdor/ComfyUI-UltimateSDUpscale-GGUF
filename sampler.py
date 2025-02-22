import os
import sys
import torch

# Add ComfyUI path to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMFY_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if COMFY_DIR not in sys.path:
    sys.path.append(COMFY_DIR)

from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
import comfy.sample
import comfy.model_management
import latent_preview

# VRAM Management
import torch.cuda
from comfy.cli_args import args
import comfy.model_management

# Get the current device
device = comfy.model_management.get_torch_device()

# Get VRAM information
total_vram, torch_total_vram = comfy.model_management.get_total_memory(device, torch_total_too=True)
free_vram, torch_free_vram = comfy.model_management.get_free_memory(device, torch_free_too=True)

# Convert to GB for better readability
total_vram_gb = total_vram / (1024 * 1024 * 1024)
free_vram_gb = free_vram / (1024 * 1024 * 1024)

# Check for reserved VRAM from command line args (--reserve-vram CLI arg becomes reserve_vram in args)
reserved_vram = getattr(args, 'reserve_vram', 0)
if reserved_vram > 0:
    # Convert MB to bytes since reserve_vram is specified in MB
    reserved_vram_bytes = reserved_vram * 1024 * 1024
    free_vram -= reserved_vram_bytes
    free_vram_gb = free_vram / (1024 * 1024 * 1024)

# Store these for use in the sampler
VRAM_TOTAL = total_vram
VRAM_FREE = free_vram
VRAM_RESERVED = reserved_vram * 1024 * 1024 if reserved_vram > 0 else 0

class SamplerHelper:
    @staticmethod
    def force_memory_cleanup(unload_models=False):
        comfy.model_management.cleanup_models()
        if unload_models:
            comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache(True)
        torch.cuda.empty_cache()

    @staticmethod
    def create_gaussian_blend_kernel(overlap, device):
        """Create a gaussian blend kernel that matches BHWC format."""
        try:
            y = torch.arange(overlap, device=device).float()
            x = torch.arange(overlap, device=device).float()
            
            # Calculate 1D gaussian weights
            center = overlap / 2
            sigma = overlap / 4  # Controls the spread of the gaussian
            
            # Create 1D gaussians
            y_kernel = torch.exp(-((y - center) ** 2) / (2 * sigma ** 2))
            x_kernel = torch.exp(-((x - center) ** 2) / (2 * sigma ** 2))
            
            # Expand to match dimensions properly
            y_kernel = y_kernel.view(-1, 1)  # Shape: [overlap, 1]
            x_kernel = x_kernel.view(1, -1)  # Shape: [1, overlap]
            
            # Create 2D kernel through outer product
            kernel = y_kernel @ x_kernel  # Shape: [overlap, overlap]
            
            # Normalize kernel
            kernel = kernel / kernel.max()
            
            # Add batch dimension and move to BHWC format
            # [overlap, overlap] -> [1, overlap, overlap, 1]
            kernel = kernel.view(1, overlap, overlap, 1)
            
            return kernel
        finally:
            # Cleanup temporary tensors
            del y, x, y_kernel, x_kernel

    @staticmethod
    def blend_tile_edges(tile, existing_output, kernel):
        """Blend tile edges using gaussian kernel. Expects BHWC format."""
        print(f"DEBUG: Initial shapes:")
        print(f"  tile: {tile.shape}")
        print(f"  existing_output: {existing_output.shape}")
        print(f"  kernel: {kernel.shape}")
        
        overlap = kernel.shape[1]  # Kernel is [B,H,W,C]
        
        # Extract overlap regions
        left = existing_output[:, :, :overlap, :] if tile.shape[2] > overlap else None
        right = existing_output[:, :, -overlap:, :] if tile.shape[2] > overlap else None
        top = existing_output[:, :overlap, :, :] if tile.shape[1] > overlap else None
        bottom = existing_output[:, -overlap:, :, :] if tile.shape[1] > overlap else None
        
        if left is not None:
            print(f"  left overlap: {left.shape}")
        if top is not None:
            print(f"  top overlap: {top.shape}")
        
        # Expand kernel to match tile dimensions
        kernel_horizontal = kernel.expand(tile.shape[0], overlap, overlap, 1)
        kernel_vertical = kernel.permute(0, 2, 1, 3).expand(tile.shape[0], overlap, overlap, 1)
        
        print(f"DEBUG: Kernel shapes after expansion:")
        print(f"  kernel_horizontal: {kernel_horizontal.shape}")
        print(f"  kernel_vertical: {kernel_vertical.shape}")
        
        # Blend edges
        if left is not None:
            # Extract the section we're blending
            section = tile[:, :, :overlap, :]
            print(f"  blending section shape: {section.shape}")
            print(f"  kernel_horizontal shape: {kernel_horizontal.shape}")
            
            # Expand kernel to match height
            kernel_h = kernel_horizontal.expand(-1, section.shape[1], -1, -1)
            tile[:, :, :overlap, :] = section * kernel_h + left * (1 - kernel_h)
            
        if right is not None:
            section = tile[:, :, -overlap:, :]
            kernel_h = kernel_horizontal.expand(-1, section.shape[1], -1, -1)
            tile[:, :, -overlap:, :] = section * kernel_h.flip(2) + right * (1 - kernel_h.flip(2))
            
        if top is not None:
            section = tile[:, :overlap, :, :]
            kernel_v = kernel_vertical.expand(-1, -1, section.shape[2], -1)
            tile[:, :overlap, :, :] = section * kernel_v + top * (1 - kernel_v)
            
        if bottom is not None:
            section = tile[:, -overlap:, :, :]
            kernel_v = kernel_vertical.expand(-1, -1, section.shape[2], -1)
            tile[:, -overlap:, :, :] = section * kernel_v.flip(1) + bottom * (1 - kernel_v.flip(1))
        
        return tile

    @staticmethod
    def process_latent_batch(latents, noise, guider, sampler, sigmas):
        processed_latents = []
        
        for latent in latents:
            # Process single latent
            processed = Sampler.sample(noise, guider, sampler, sigmas, latent)
            processed_latents.append(processed)
            
            # Clear current latent
            del latent
        
        # Aggressive memory cleanup after function completes
        SamplerHelper.force_memory_cleanup()
        
        return processed_latents

class OptimizedSampler:
    def __init__(self):
        self.last_samples = None
        self.callback_count = 0
    
    def sample(self, noise, guider, sampler, sigmas, latent_image):
        # Process latent
        latent_image["samples"] = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image["samples"])
        
        # Handle noise mask
        noise_mask = None
        if "noise_mask" in latent_image:
            noise_mask = latent_image["noise_mask"]
        
        # Setup callback for progress
        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
        
        try:
            # Generate noise
            noise_tensor = noise.generate_noise(latent_image)
            
            # Sample
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
            samples = guider.sample(
                noise_tensor, 
                latent_image["samples"], 
                sampler, 
                sigmas, 
                denoise_mask=noise_mask, 
                callback=callback, 
                disable_pbar=disable_pbar,
                seed=noise.seed
            )
            
            return {"samples": samples}
            
        finally:
            # Just cleanup tensors we created
            if 'noise_tensor' in locals():
                del noise_tensor
            if noise_mask is not None:
                del noise_mask

class Sampler:
    @staticmethod
    def sample(noise, guider, sampler, sigmas, latent):
        # Create optimized sampler instance
        opt_sampler = OptimizedSampler()
        
        # Sample using optimized sampler
        samples = opt_sampler.sample(noise, guider, sampler, sigmas, latent)
        del opt_sampler
        
        # Package output
        output = {"samples": samples["samples"]}
        if "noise_mask" in latent:
            output["noise_mask"] = latent["noise_mask"]
        
        return output

    @staticmethod
    def encode(image, vae):
        t = vae.encode(image[:,:,:,:3])
        return {"samples": t}
