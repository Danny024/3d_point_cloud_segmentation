import os
import cv2
from modules.data_processing import PointCloudProcessor
from modules.sam import SAMSegmenter
from modules.visualize import ImageVisualizer
import numpy as np

class Pipeline:
    """Orchestrates point cloud processing, segmentation, and visualization."""
    
    def __init__(self, pcd_path: str, output_dir: str, resolution: int = 500):
        """
        Initialize pipeline with file paths and settings.
        
        Args:
            pcd_path (str): Path to input .las file.
            output_dir (str): Directory for output files.
            resolution (int): Vertical resolution for spherical projection.
        """
        self.pcd_path = pcd_path
        self.output_dir = output_dir
        self.resolution = resolution
        self.processor = PointCloudProcessor(pcd_path, resolution)
        self.segmenter = SAMSegmenter()
        self.visualizer = ImageVisualizer()
        
        os.makedirs(output_dir, exist_ok=True)

    def run(self):
        """Execute the processing pipeline."""
        # Generate spherical projection
        center_coordinates = self.processor.get_center_coordinates()
        spherical_image, mapping = self.processor.generate_spherical_image(center_coordinates)
        
        # Save spherical projection
        sphere_path = os.path.join(self.output_dir, "sphere_projection.jpg")
        self.visualizer.plot_image(spherical_image, sphere_path)
        
        # Generate masks with SAM
        temp_img = cv2.imread(sphere_path)
        image_rgb = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
        masks = self.segmenter.generate_masks(image_rgb)
        
        # Save segmented image
        segmented_path = os.path.join(self.output_dir, "2d_segmented_projection.jpg")
        self.visualizer.plot_image(image_rgb, segmented_path, masks=masks)
        
        # Color and export point cloud
        modified_point_cloud = self.processor.color_point_cloud(segmented_path, mapping)
        segmented_pcd = os.path.join(self.output_dir, "3d_segmented_point_cloud.las")
        self.processor.export_point_cloud(segmented_pcd, modified_point_cloud)

if __name__ == "__main__":
    HOME = os.getcwd()
    pipeline = Pipeline(
        pcd_path=os.path.join(HOME, "data", "unreal.las"),
        output_dir=os.path.join(HOME, "result"),
        resolution=500
    )
    pipeline.run()