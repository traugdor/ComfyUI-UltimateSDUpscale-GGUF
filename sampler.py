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
        
        # Package output
        output = {"samples": samples["samples"]}
        if "noise_mask" in latent:
            output["noise_mask"] = latent["noise_mask"]
        
        return output

    @staticmethod
    def encode(image, vae):
        t = vae.encode(image[:,:,:,:3])
        return {"samples": t}
