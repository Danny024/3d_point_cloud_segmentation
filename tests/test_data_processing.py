import unittest
import numpy as np
from unittest.mock import patch, Mock
from modules.data_processing import PointCloudProcessor

class TestPointCloudProcessor(unittest.TestCase):
    def setUp(self):
        """Set up mock data and processor instance."""
        self.pcd_path = "mock.las"
        self.resolution_y = 100
        self.mock_las = Mock()
        self.mock_las.x = np.array([0, 1, 2])
        self.mock_las.y = np.array([0, 1, 2])
        self.mock_las.z = np.array([0, 1, 2])
        self.mock_las.red = np.array([65535, 32768, 0])
        self.mock_las.green = np.array([0, 32768, 65535])
        self.mock_las.blue = np.array([65535, 0, 32768])
        self.processor = PointCloudProcessor(self.pcd_path, self.resolution_y)

    @patch('laspy.read')
    def test_load_point_cloud(self, mock_read):
        """Test point cloud loading and color extraction."""
        mock_read.return_value = self.mock_las
        processor = PointCloudProcessor(self.pcd_path, self.resolution_y)
        expected_points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        expected_colors = np.array([[255, 0, 255], [128, 128, 0], [0, 255, 128]])
        np.testing.assert_array_equal(processor.point_cloud, expected_points)
        np.testing.assert_array_equal(processor.colors, expected_colors)

    @patch('laspy.read')
    def test_get_center_coordinates(self, mock_read):
        """Test center coordinate computation."""
        mock_read.return_value = self.mock_las
        processor = PointCloudProcessor(self.pcd_path, self.resolution_y)
        center = processor.get_center_coordinates()
        expected_center = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(center, expected_center)

    def test_generate_spherical_image(self):
        """Test spherical image generation with mock points."""
        self.processor.point_cloud = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.processor.colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
        center = np.array([0, 0, 0])
        image, mapping = self.processor.generate_spherical_image(center)
        self.assertEqual(image.shape, (self.resolution_y, 2 * self.resolution_y, 3))
        self.assertEqual(mapping.shape, (self.resolution_y, 2 * self.resolution_y))
        self.assertTrue(np.any(image != 0))  # Some pixels should be colored

    @patch('cv2.imread')
    def test_color_point_cloud(self, mock_imread):
        """Test coloring point cloud with mock image."""
        mock_image = np.zeros((50, 100, 3), dtype=np.uint8)
        mock_image[25, 50] = [255, 128, 64]
        mock_imread.return_value = mock_image
        self.processor.point_cloud = np.array([[1, 0, 0]])
        mapping = np.zeros((100, 200), dtype=int)
        mapping[50, 100] = 0
        modified_cloud = self.processor.color_point_cloud("mock.jpg", mapping)
        expected_cloud = np.array([[1, 0, 0, 255, 128, 64]], dtype=np.float32)
        np.testing.assert_array_almost_equal(modified_cloud, expected_cloud)

    @patch('laspy.LasData')
    @patch('laspy.LasHeader')
    def test_export_point_cloud(self, mock_header, mock_las_data):
        """Test exporting point cloud with mock LAS data."""
        mock_header_instance = Mock()
        mock_header.return_value = mock_header_instance
        mock_las_instance = Mock()
        mock_las_data.return_value = mock_las_instance
        cloud = np.array([[1, 2, 3, 255, 128, 64]], dtype=np.float32)
        self.processor.export_point_cloud("output.las", cloud)
        mock_header_instance.add_extra_dim.assert_called_once()
        mock_las_instance.write.assert_called_once_with("output.las")

if __name__ == '__main__':
    unittest.main()