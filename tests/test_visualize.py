import unittest
import numpy as np
from unittest.mock import patch, Mock
from modules.visualize import ImageVisualizer

class TestImageVisualizer(unittest.TestCase):
    def setUp(self):
        """Set up mock image and masks."""
        self.visualizer = ImageVisualizer()
        self.image = np.zeros((100, 200, 3), dtype=np.uint8)
        self.masks = [{'segmentation': np.ones((100, 200), dtype=bool), 'area': 1000}]

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.gca')
    @patch('matplotlib.pyplot.imshow')
    @patch('matplotlib.pyplot.axis')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_image_without_masks(self, mock_close, mock_savefig, mock_axis, mock_imshow, mock_gca, mock_figure):
        """Test plotting image without masks."""
        save_path = "test.jpg"
        self.visualizer.plot_image(self.image, save_path)
        mock_figure.assert_called_with(figsize=(200 / 72, 100 / 72))
        mock_axis.assert_called_with('off')
        mock_savefig.assert_called_with(save_path)
        mock_close.assert_called_once()

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.gca')
    @patch('matplotlib.pyplot.imshow')
    @patch('matplotlib.pyplot.axis')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_image_with_masks(self, mock_close, mock_savefig, mock_axis, mock_imshow, mock_gca, mock_figure):
        """Test plotting image with masks."""
        save_path = "test.jpg"
        mock_ax = Mock()
        mock_gca.return_value = mock_ax
        self.visualizer.plot_image(self.image, save_path, masks=self.masks)
        mock_figure.assert_called_with(figsize=(200 / 72, 100 / 72))
        mock_ax.set_autoscale_on.assert_called_with(False)
        self.assertEqual(mock_imshow.call_count, 2)  # Image + mask
        mock_axis.assert_called_with('off')
        mock_savefig.assert_called_with(save_path)
        mock_close.assert_called_once()

if __name__ == '__main__':
    unittest.main()