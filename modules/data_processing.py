import numpy as np
import laspy
import cv2
import os

class PointCloudProcessor:
    """Helps handle point cloud loading, transformation, coloring, and exporting."""
    
    def __init__(self, pcd_path: str, resolution_y: int = 500):
        """
        Initialize with point cloud path and resolution.
        
        Args:
            pcd_path (str): Path to the .las point cloud file.
            resolution_y (int): Vertical resolution for spherical projection.
        """
        self.pcd_path = pcd_path
        self.resolution_y = resolution_y
        self.point_cloud = None
        self.colors = None
        self._load_point_cloud()

    def _load_point_cloud(self):
        """Load point cloud and extract coordinates and colors."""
        las = laspy.read(self.pcd_path)
        self.point_cloud = np.vstack((las.x, las.y, las.z)).transpose()
        self.colors = np.vstack((
            (las.red / 65535 * 255).astype(int),
            (las.green / 65535 * 255).astype(int),
            (las.blue / 65535 * 255).astype(int)
        )).transpose()

    def get_center_coordinates(self) -> np.ndarray:
        """
        helps to get center coordinates of the point cloud.
        
        Returns:
            np.ndarray: Center coordinates [x, y, z].
        """
        las = laspy.read(self.pcd_path)
        x, y, z = las.x, las.y, las.z
        return np.array([
            (x.min() + x.max()) / 2,
            (y.min() + y.max()) / 2,
            (z.min() + z.max()) / 2
        ])

    def generate_spherical_image(self, center_coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates spherical projection image and mapping.
        
        Args:
            center_coordinates (np.ndarray): Center coordinates [x, y, z].
        
        Returns:
            tuple: (spherical image, mapping array).
        """
        translated_points = self.point_cloud - center_coordinates
        theta = np.arctan2(translated_points[:, 1], translated_points[:, 0])
        phi = np.arccos(translated_points[:, 2] / np.linalg.norm(translated_points, axis=1))
        
        x = (theta + np.pi) / (2 * np.pi) * (2 * self.resolution_y)
        y = phi / np.pi * self.resolution_y
        
        resolution_x = 2 * self.resolution_y
        image = np.zeros((self.resolution_y, resolution_x, 3), dtype=np.uint8)
        mapping = np.full((self.resolution_y, resolution_x), -1, dtype=int)
        
        for i in range(len(translated_points)):
            ix = np.clip(int(x[i]), 0, resolution_x - 1)
            iy = np.clip(int(y[i]), 0, self.resolution_y - 1)
            if mapping[iy, ix] == -1 or np.linalg.norm(translated_points[i]) < np.linalg.norm(translated_points[mapping[iy, ix]]):
                mapping[iy, ix] = i
                image[iy, ix] = self.colors[i]
        
        return image, mapping

    def color_point_cloud(self, image_path: str, mapping: np.ndarray) -> np.ndarray:
        """
        Color point cloud based on segmented image.
        
        Args:
            image_path (str): Path to segmented image.
            mapping (np.ndarray): Mapping from image to point cloud.
        
        Returns:
            np.ndarray: Modified point cloud with colors.
        """
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        scale_y = mapping.shape[0] / h
        scale_x = mapping.shape[1] / w
        
        modified_point_cloud = np.zeros((self.point_cloud.shape[0], self.point_cloud.shape[1] + 3), dtype=np.float32)
        modified_point_cloud[:, :3] = self.point_cloud
        
        for iy in range(h):
            for ix in range(w):
                mapped_ix = int(ix * scale_x)
                mapped_iy = int(iy * scale_y)
                point_index = mapping[mapped_iy, mapped_ix]
                if point_index != -1:
                    modified_point_cloud[point_index, 3:] = image[iy, ix]
        
        return modified_point_cloud

    def export_point_cloud(self, output_path: str, modified_point_cloud: np.ndarray):
        """
        Exports colored point cloud to .las file.
        
        Args:
            output_path (str): Path to save the .las file.
            modified_point_cloud (np.ndarray): Point cloud with colors.
        """
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.add_extra_dim(laspy.ExtraBytesParams(name="random", type=np.int32))
        
        las_o = laspy.LasData(header)
        las_o.x = modified_point_cloud[:, 0]
        las_o.y = modified_point_cloud[:, 1]
        las_o.z = modified_point_cloud[:, 2]
        las_o.red = modified_point_cloud[:, 3]
        las_o.green = modified_point_cloud[:, 4]
        las_o.blue = modified_point_cloud[:, 5]
        las_o.write(output_path)
        
        print(f"Export successful at: {output_path}")