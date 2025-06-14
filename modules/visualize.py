import matplotlib.pyplot as plt
import numpy as np

class ImageVisualizer:
    """Handles plotting of images with optional masks."""
    
    def plot_image(self, image: np.ndarray, save_path: str, masks: list = None):
        """
        Plots an image with optional masks and save to disk.
        
        Args:
            image (np.ndarray): Image array to plot.
            save_path (str): Path to save the plot.
            masks (list, optional): List of mask dictionaries.
        """
        fig = plt.figure(figsize=(image.shape[1] / 72, image.shape[0] / 72))
        fig.add_axes([0, 0, 1, 1])
        
        plt.imshow(image)
        
        if masks:
            if len(masks) > 0:
                sorted_anns = sorted(masks, key=lambda x: x['area'], reverse=True)
                ax = plt.gca()
                ax.set_autoscale_on(False)
                
                for ann in sorted_anns:
                    m = ann['segmentation']
                    img = np.ones((m.shape[0], m.shape[1], 3))
                    color_mask = np.random.random((1, 3)).tolist()[0]
                    for i in range(3):
                        img[:, :, i] = color_mask[i]
                    ax.imshow(np.dstack((img, m * 0.8)))
        
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()