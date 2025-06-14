import unittest
from unittest.mock import patch, Mock
import numpy as np
from modules.sam import SAMSegmenter

class TestSAMSegmenter(unittest.TestCase):
    def setUp(self):
        """Set up mock checkpoint path."""
        self.checkpoint_path = "mock.pth"

    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.cuda.empty_cache')
    @patch('segment_anything.sam_model_registry')
    def test_initialize_sam(self, mock_registry, mock_empty_cache, mock_cuda):
        """Test SAM model initialization on CPU."""
        mock_model = Mock()
        mock_registry.__getitem__.return_value = mock_model
        segmenter = SAMSegmenter(self.checkpoint_path)
        mock_registry.__getitem__.assert_called_with("vit_h")
        mock_model.assert_called_with(checkpoint=self.checkpoint_path)
        self.assertEqual(segmenter.device.type, 'cpu')
        mock_model.return_value.to.assert_called_with(device=segmenter.device)

    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch.cuda.empty_cache')
    @patch('segment_anything.sam_model_registry')
    @patch('segment_anything.SamAutomaticMaskGenerator')
    def test_generate_masks(self, mock_generator, mock_registry, mock_empty_cache, mock_cuda):
        """Test mask generation with mock SAM model."""
        mock_model = Mock()
        mock_registry.__getitem__.return_value = mock_model
        mock_model.return_value.to.return_value = mock_model
        mock_mask_gen = Mock()
        mock_generator.return_value = mock_mask_gen
        mock_mask_gen.generate.return_value = [{'segmentation': np.ones((100, 200), dtype=bool), 'area': 1000}]
        segmenter = SAMSegmenter(self.checkpoint_path)
        image_rgb = np.zeros((100, 200, 3), dtype=np.uint8)
        masks = segmenter.generate_masks(image_rgb)
        mock_generator.assert_called_with(segmenter.sam)
        mock_mask_gen.generate.assert_called_with(image_rgb)
        self.assertEqual(len(masks), 1)
        self.assertTrue(np.array_equal(masks[0]['segmentation'], np.ones((100, 200), dtype=bool)))

if __name__ == '__main__':
    unittest.main()