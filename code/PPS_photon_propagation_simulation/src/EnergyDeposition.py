import numpy as np
import matplotlib.pyplot as plt
import os
import sys

import src.PixelTranslator as PixelTranslator


class Edep_analyser:
    """
    Analyzes energy deposition data from a simulation.
    Simulation is about 2D photon propagation though soft tissue.
    Returns a matrix of energy deposition values in pixels, normalized (max values equals 1).
    """
    def __init__(self, filepath, num_pixels_x, num_pixels_y, detector_shape):
        self.filepath = filepath
        self.n_pixels_x = num_pixels_x
        self.n_pixels_y = num_pixels_y
        self.detector_shape = detector_shape
        self.energy_deposition = np.zeros((num_pixels_x, num_pixels_y))
        self.pixel_translator = PixelTranslator.pixel_translator(detector_shape[0], detector_shape[1], num_pixels_x, num_pixels_y)
        self.Edep_map = np.zeros((num_pixels_x, num_pixels_y))
        self.Edep_map_normalized = np.zeros((num_pixels_x, num_pixels_y))

    def analyse(self):
        self.load_energy_deposition()
        self.normalize(threshold_ratio=0.01)
        return self.Edep_map_normalized

    def load_energy_deposition(self):
        #with open(f"{simulation_data_path}/{self.filename}.txt", "r") as f:
        with open(f"{self.filepath}", "r") as f:
            for line in f:
                _,_, Edep_str, x_str, y_str = line.strip().split()
                Edep = float(Edep_str)
                x = float(x_str)
                y = float(y_str)
                i_pixel, j_pixel = self.pixel_translator.translate((x, y))
                self.energy_deposition[i_pixel, j_pixel] += Edep
        self.Edep_map = self.energy_deposition.T[::-1][:]

    def normalize(self, threshold_ratio=0.01):
        matrix = self.Edep_map
        max_val = np.max(matrix)
        threshold = threshold_ratio * max_val
        result = np.where((matrix < threshold) | (matrix < 0), 0, matrix)
        self.Edep_map_normalized = result / max_val

    def plot(self, normalized=True,title = 'Energy Deposition Map'):
        data = self.energy_deposition.T
        if normalized:
            data /= np.max(data)  # Normalize the data for plotting
        a, b = self.detector_shape[0] / 2, self.detector_shape[1] / 2
        plt.figure(figsize=(7, 5))
        plt.imshow(data, origin='lower', cmap='inferno', extent=(-a, a, -b, b))
        plt.colorbar(label='Deposited Energy (MeV)')
        plt.xlabel('X (cm)')
        plt.ylabel('Y (cm)')
        plt.title(title)
        plt.tight_layout()
        plt.show()