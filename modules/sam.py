import torch
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import numpy as np


class SAMSegmenter:
    """Manages SAM model initialization and mask generation."""
    
    def __init__(self, checkpoint_path: str = None):
        """
        Initialize SAM model.
        
        Args:
            checkpoint_path (str, optional): Path to SAM checkpoint. Defaults to weights/sam_vit_h_4b8939.pth.
        """
        self.checkpoint_path = checkpoint_path or os.path.join(os.getcwd(), "weights", "sam_vit_h_4b8939.pth")
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.sam = None
        self._initialize_sam()

    def _initialize_sam(self):
        """Load and initialize SAM model."""
        torch.cuda.empty_cache()
        self.sam = sam_model_registry["vit_h"](checkpoint=self.checkpoint_path)
        self.sam.to(device=self.device)

    def generate_masks(self, image_rgb: np.ndarray) -> list:
        """
        Generate masks for an RGB image using SAM.
        
        Args:
            image_rgb (np.ndarray): RGB image array.
        
        Returns:
            list: List of mask dictionaries.
        """
        mask_generator = SamAutomaticMaskGenerator(self.sam)
        return mask_generator.generate(image_rgb)