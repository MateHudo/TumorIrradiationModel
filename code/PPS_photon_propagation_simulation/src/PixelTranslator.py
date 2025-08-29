import numpy as np

class pixel_translator:
    def __init__(self, detector_length, detector_width, n_pixels_x, n_pixels_y):
        """
        Initialize the PixelTranslator for a 2D detector.

        Args:
            detector_length (float): Length of the detector (cm).
            detector_width (float): Width of the detector (cm).
            n_pixels_x (int): Number of pixels along the x (length) direction.
            n_pixels_y (int): Number of pixels along the y (width) direction.
        """
        self.detector_length = detector_length
        self.detector_width = detector_width
        self.n_pixels_x = n_pixels_x
        self.n_pixels_y = n_pixels_y
        self.pixel_size_x = detector_length / n_pixels_x
        self.pixel_size_y = detector_width / n_pixels_y

    def translate(self, loc):
        """
        Translate a (x, y) location to pixel indices (i, j).

        Args:
            loc (tuple): Location (x, y) in cm.

        Returns:
            tuple: Pixel indices (i, j).
        """
        x, y = loc
        # Shift origin to bottom-left corner
        i = int(np.floor((x + self.detector_length / 2) / self.pixel_size_x))
        j = int(np.floor((y + self.detector_width / 2) / self.pixel_size_y))
        # Clamp indices to valid range
        i = min(max(i, 0), self.n_pixels_x - 1)
        j = min(max(j, 0), self.n_pixels_y - 1)
        return i, j
   