import numpy as np

class pencil_beam_gps:
    """
    Pencil beam shape. 
    Beam enters at the edge center of chosen TissueMatrix cell. 
    """
    def __init__(self, tissue_shape, spectrum_type="mono", **kwargs):
        self.spectrum_type = spectrum_type
        self.tissue_length, self.tissue_width = tissue_shape["dimensions"]
        self.num_pixels_x, self.num_pixels_y = tissue_shape["num_pixels"]
        # fixme : True, each pixel same size, but condition here is wrong
        # #* each pixel shall be the same size
        # if self.tissue_length % self.num_pixels_x != 0 or self.tissue_width % self.num_pixels_y != 0:
        #     print(f"value: {self.tissue_length} % {self.num_pixels_x} = {self.tissue_length % self.num_pixels_x}")
        #     raise ValueError("Tissue dimensions must be divisible by the number of pixels.")
        self.pixel_length_x = self.tissue_length / self.num_pixels_x
        self.pixel_length_y = self.tissue_width / self.num_pixels_y
        self.pixel_dimensions = (self.pixel_length_x, self.pixel_length_y)
        #* we want odd number of pixels
        if self.num_pixels_x % 2 == 0 or self.num_pixels_y % 2 == 0:
            raise ValueError("Number of pixels must be odd to have a center pixel."
                            "\n (we want to have center pixel in order to go from 4N simulations)"
                            "to N/2 + 0.5 simulations needed to define all SITCFs!")
        # Store extra spectrum parameters
        self.spectrum_params = kwargs

    def get_gps_template(self, beam_index):
        if beam_index < 0 or beam_index >= self.num_pixels_y:
            raise ValueError("Error: beam_index out of range for simulating from left side!")
        x_entrance = - self.tissue_length / 2 + 10**-10  # entrance at left edge, small offset to avoid boundary issues
        y_entrance = self.tissue_width / 2 - self.pixel_length_y * (beam_index + 0.5)
        r_entrance = np.array([x_entrance, y_entrance, 0])
        u_entrance = np.array([1, 0, 0])

        result = {
            "spectrum_type": self.spectrum_type,
            "beam_index": beam_index,
            "r": r_entrance,
            "u": u_entrance,
        }

        if self.spectrum_type == "mono":
            # Use Emono from kwargs if provided, else default
            result["Emono"] = self.spectrum_params.get("Emono", 0.662)
        elif self.spectrum_type == "rtg":
            # Require Emax, alpha, beta in kwargs
            try:
                result["Emax"] = self.spectrum_params["Emax"]
                result["alpha"] = self.spectrum_params["alpha"]
                result["beta"] = self.spectrum_params["beta"]
            except KeyError as e:
                raise ValueError(f"Missing required parameter for 'rtg' spectrum: {e}")
        else:
            raise ValueError("Error: Unknown spectrum_type. Use 'mono' or 'rtg'.")

        return result






def isotropic_gps(tissue_dimensions, type="mono"):
    ## space for development

    return {
        "type": type
    }

# def ...
