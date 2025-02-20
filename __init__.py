"""
ComfyUI Ultimate SD Upscale Node for GGUF models
"""

from .ultimate_sd_upscale_gguf import UltimateSDUpscaleGGUF

NODE_CLASS_MAPPINGS = {
    "UltimateSDUpscaleGGUF": UltimateSDUpscaleGGUF
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UltimateSDUpscaleGGUF": "Ultimate SD Upscale (GGUF)"
}
