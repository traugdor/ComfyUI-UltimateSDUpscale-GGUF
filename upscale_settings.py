import math

class UpscaleSettings:
    MIN_TILE_SIZE = 256
    MINIMUM_BENEFICIAL_UPSCALE = 1.5
    NEARLY_SQUARE_THRESHOLD = 0.01  # 1% difference threshold
    
    def __init__(self, target_width, target_height, max_tile_size, tile_padding, force_uniform_tiles=False):
        self.target_width = target_width
        self.target_height = target_height
        self.tile_padding = tile_padding
        self.force_uniform = force_uniform_tiles
        self.max_tile_size = max_tile_size
        
        # Calculate final tile dimensions
        self.tile_width, self.tile_height, self.sampling_width, self.sampling_height = self._calculate_tile_dimensions()
    
    def _calculate_uniform_tile_dimensions(self):
        # Calculate initial number of tiles
        # x = width, y = height
        # find longest side first and divide by max tile size
        if self.target_width > self.target_height:
            self.num_tiles_x = math.ceil(self.target_width / self.max_tile_size)
            max_tile_width = self.max_tile_size
            max_tile_height = math.ceil(self.max_tile_size * (self.target_height / self.target_width) // 8 ) * 8
            self.num_tiles_y = math.ceil(self.target_height / max_tile_height)
        else:
            self.num_tiles_y = math.ceil(self.target_height / self.max_tile_size)
            max_tile_width = math.ceil(self.max_tile_size * (self.target_width / self.target_height) // 8 ) * 8
            max_tile_height = self.max_tile_size
            self.num_tiles_x = math.ceil(self.target_width / max_tile_width)

        sampling_width = self.num_tiles_x * max_tile_width
        sampling_height = self.num_tiles_y * max_tile_height
        
        return max_tile_width, max_tile_height, sampling_width, sampling_height

    def _calculate_tile_dimensions(self):
        # consume self.tile_ratio
        if self.force_uniform:
            tile_width, tile_height, sampling_width, sampling_height = self._calculate_uniform_tile_dimensions()
            if tile_width is not None and tile_height is not None:
                return tile_width, tile_height, sampling_width, sampling_height

        # Double check that image dimensions are a multiple of 8 and adjust if necessary
        # Return as sampling dimensions
        sampling_width = self.target_width
        sampling_height = self.target_height
        if self.target_width % 8 != 0:
            sampling_width  = (self.target_width // 8 + 1) * 8
        if self.target_height % 8 != 0:
            sampling_height = (self.target_height // 8 + 1) * 8
        
        # For non-uniform tiles, use square tiles
        tile_size = max(64, math.ceil(self.max_tile_size / 8) * 8)
        tile_size = min(tile_size, min(self.target_width, self.target_height))
        
        # Calculate number of full tiles needed
        self.num_tiles_x = math.ceil(self.target_width / tile_size)
        self.num_tiles_y = math.ceil(self.target_height / tile_size)
        
        return tile_size, tile_size, sampling_width, sampling_height
    
    def get_tile_coordinates(self, tile_x, tile_y, tile_padding):
        if not (0 <= tile_x < self.num_tiles_x and 0 <= tile_y < self.num_tiles_y):
            return None, None, None, None
            
        # Calculate base tile coordinates without padding
        x1 = tile_x * self.tile_width
        x2 = min((tile_x + 1) * self.tile_width, self.target_width)
        y1 = tile_y * self.tile_height
        y2 = min((tile_y + 1) * self.tile_height, self.target_height)
        
        # Add padding only at seams
        pad_left = tile_padding if tile_x > 0 else 0
        pad_right = tile_padding if tile_x < self.num_tiles_x - 1 else 0
        pad_top = tile_padding if tile_y > 0 else 0
        pad_bottom = tile_padding if tile_y < self.num_tiles_y - 1 else 0
        
        # Apply padding to coordinates
        pad_x1 = max(0, x1 - pad_left)
        pad_x2 = min(self.target_width, x2 + pad_right)
        pad_y1 = max(0, y1 - pad_top)
        pad_y2 = min(self.target_height, y2 + pad_bottom)
        
        return x1, x2, y1, y2, pad_x1, pad_x2, pad_y1, pad_y2

    def get_tile_mask_area(self, tile_width, tile_height, tile_padding):
        tile_start_x = max(0, tile_padding)
        tile_end_x = min(tile_width, self.tile_width + tile_padding)
        tile_start_y = max(0, tile_padding)
        tile_end_y = min(tile_height, self.tile_height + tile_padding)
        
        return tile_start_x, tile_end_x, tile_start_y, tile_end_y
