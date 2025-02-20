import math

class UpscaleSettings:
    MIN_TILE_SIZE = 256
    MINIMUM_BENEFICIAL_UPSCALE = 1.5
    NEARLY_SQUARE_THRESHOLD = 0.01  # 1% difference threshold
    
    def __init__(self, original_width, original_height, upscale_by, tile_width, tile_height, force_uniform_tiles):
        self.original_width = original_width
        self.original_height = original_height
        self.upscale_by = upscale_by
        self.base_tile_width = tile_width
        self.base_tile_height = tile_height
        self.force_uniform = force_uniform_tiles
        
        # Calculate target dimensions
        self.target_width = int(original_width * upscale_by)
        self.target_height = int(original_height * upscale_by)
        
        # Determine if we should use uniform tiles
        self.use_uniform_tiles = self._should_use_uniform_tiles()
        
        # Calculate actual tile dimensions
        self.tile_width, self.tile_height = self._calculate_tile_dimensions()
        
        # Calculate number of tiles
        self.num_tiles_x, self.num_tiles_y = self._calculate_num_tiles()
    
    def _is_nearly_square(self):
        """Check if the target dimensions are nearly square"""
        aspect_ratio = self.target_width / self.target_height
        return 1 - self.NEARLY_SQUARE_THRESHOLD <= aspect_ratio <= 1 + self.NEARLY_SQUARE_THRESHOLD
    
    def _is_upscale_beneficial(self):
        """Check if the upscale factor is large enough to benefit from uniform tiles"""
        return self.upscale_by >= self.MINIMUM_BENEFICIAL_UPSCALE
    
    def _should_use_uniform_tiles(self):
        """Determine if uniform tiles should be used based on image characteristics"""
        if not self.force_uniform:
            return False
        
        if self._is_nearly_square():
            return False
            
        if not self._is_upscale_beneficial():
            return False
            
        return True
    
    def _calculate_uniform_tile_dimensions(self):
        """Calculate dimensions for uniform tiles"""
        # Calculate number of tiles needed
        width_tiles = math.ceil(self.target_width / self.base_tile_width)
        height_tiles = math.ceil(self.target_height / self.base_tile_height)
        
        # Calculate actual tile dimensions
        tile_width = self.target_width / width_tiles
        tile_height = self.target_height / height_tiles
        
        # Round to nearest multiple of 8 for latent space
        tile_width = math.ceil(tile_width / 8) * 8
        tile_height = math.ceil(tile_height / 8) * 8
        
        # Check if tiles would be too small
        if tile_width < self.MIN_TILE_SIZE or tile_height < self.MIN_TILE_SIZE:
            return None, None
            
        return tile_width, tile_height
    
    def _calculate_tile_dimensions(self):
        """Calculate final tile dimensions based on settings"""
        if self.use_uniform_tiles:
            tile_width, tile_height = self._calculate_uniform_tile_dimensions()
            if tile_width is not None and tile_height is not None:
                return tile_width, tile_height
        
        return self.base_tile_width, self.base_tile_height
    
    def _calculate_num_tiles(self):
        """Calculate number of tiles needed in each dimension"""
        return (
            math.ceil(self.target_width / self.tile_width),
            math.ceil(self.target_height / self.tile_height)
        )
    
    def get_tile_coordinates(self, tile_x, tile_y, tile_padding):
        """Calculate coordinates for a specific tile including padding"""
        if self.use_uniform_tiles:
            start_x = int(tile_x * self.tile_width - (tile_padding if tile_x > 0 else 0))
            end_x = int(min((tile_x + 1) * self.tile_width + (tile_padding if tile_x < self.num_tiles_x - 1 else 0), self.target_width))
            start_y = int(tile_y * self.tile_height - (tile_padding if tile_y > 0 else 0))
            end_y = int(min((tile_y + 1) * self.tile_height + (tile_padding if tile_y < self.num_tiles_y - 1 else 0), self.target_height))
        else:
            start_x = max(0, tile_x * self.tile_width - tile_padding)
            end_x = min(self.target_width, (tile_x + 1) * self.tile_width + tile_padding)
            start_y = max(0, tile_y * self.tile_height - tile_padding)
            end_y = min(self.target_height, (tile_y + 1) * self.tile_height + tile_padding)
        
        return start_x, end_x, start_y, end_y
    
    def get_tile_mask_area(self, tile_width, tile_height, tile_padding):
        """Calculate the actual area to be processed within a tile"""
        tile_start_x = max(0, tile_padding)
        tile_end_x = min(tile_width, self.tile_width + tile_padding)
        tile_start_y = max(0, tile_padding)
        tile_end_y = min(tile_height, self.tile_height + tile_padding)
        
        return tile_start_x, tile_end_x, tile_start_y, tile_end_y
