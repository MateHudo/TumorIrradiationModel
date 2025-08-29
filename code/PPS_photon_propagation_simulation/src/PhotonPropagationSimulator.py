import os
import numpy as np

class photon_propagation_simulator:
    def __init__(
        self,
        tissue_length,
        num_pixels,
        spectrum_type,
        Nsim,
        LAC_data_tissue,
        Simulator,
        gps,
        Emono=None,
        Emax=None,
        alpha=None,
        beta=None,
        Emin_terminate=0.001,
        output_dir="./simulation_data",
        SIF_dir="../../SIF_coefficients",
        tissue_lac_importer="none",
    ):
        self.tissue_length = tissue_length
        self.num_pixels = num_pixels
        self.spectrum_type = spectrum_type
        self.Nsim = Nsim
        self.LAC_data_tissue = LAC_data_tissue
        self.tissue_lac_importer = tissue_lac_importer
        self.Simulator = Simulator
        self.gps = gps
        self.EMIN_TERMINATE = Emin_terminate
        self.output_dir = output_dir
        self.SIF_dir = SIF_dir

        # Beam parameters
        self.Emono = Emono
        self.Emax = Emax
        self.alpha = alpha
        self.beta = beta

        # Derived parameters
        self.tissue_box = (tissue_length, tissue_length)
        self.pixel_size = (tissue_length / num_pixels, tissue_length / num_pixels)
        self.tissue_shape = {
            "dimensions": self.tissue_box,
            "num_pixels": (num_pixels, num_pixels)
        }
        if num_pixels % 2 == 0:
            raise ValueError("Number of pixels must be odd (square matrix)!")
        self.max_beam_index = num_pixels // 2
        self.Nupper = self.max_beam_index + 1

        # Output folder and file names
        self.Ebeam = self._get_Ebeam_label()
        self.foldername = f"{self.spectrum_type}_{self.tissue_length}cm_{self.num_pixels}pixels_{self.Ebeam}_{self.Nsim:.0e}photons"
        self.subfolder_name = os.path.join(self.output_dir, self.foldername)
        self.sif_filename = f"SIF_{self.foldername}"
        self.SIF_filepath = os.path.join(self.SIF_dir, f"{self.sif_filename}.npy")

        # Prepare simulator and beam GPS
        self.simulator = self.Simulator.box_propagation_simulator2D(
            box_dimensions=self.tissue_box,
            LAC_data=self.LAC_data_tissue,
            Emin_terminate=self.EMIN_TERMINATE
        )
        self.pencil_beam_gps = self._init_beam_gps()

    def _get_Ebeam_label(self):
        if self.spectrum_type == "mono":
            if self.Emono is None:
                raise ValueError("Emono must be set for monoenergetic spectrum.")
            return f"{self.Emono*1000:.0f}kev"
        elif self.spectrum_type == "rtg":
            if self.Emax is None:
                raise ValueError("Emax must be set for RTG spectrum.")
            return f"{self.Emax:.0f}MV"
        else:
            raise ValueError("Unknown spectrum_type.")

    def _init_beam_gps(self):
        if self.spectrum_type == "mono":
            return self.gps.pencil_beam_gps(
                tissue_shape=self.tissue_shape,
                spectrum_type=self.spectrum_type,
                Emono=self.Emono
            )
        elif self.spectrum_type == "rtg":
            return self.gps.pencil_beam_gps(
                tissue_shape=self.tissue_shape,
                spectrum_type=self.spectrum_type,
                Emax=self.Emean,
                alpha=self.alpha,
                beta=self.beta
            )
        else:
            raise ValueError("Unknown spectrum_type.")

    @property
    def Emean(self):
        if self.spectrum_type == "rtg":
            return self.Emax / 3
        elif self.spectrum_type == "mono":
            return self.Emono
        else:
            raise ValueError("Unknown spectrum_type.")

    def run_simulation(self):
        os.makedirs(self.subfolder_name, exist_ok=True)
        output_mapping = {"on": True, "folder_name": self.subfolder_name}
        for beam_index in range(self.Nupper):
            gps_template = self.pencil_beam_gps.get_gps_template(beam_index=beam_index)
            output_mapping["file_name"] = f"beam{beam_index}"
            self.simulator.simulate(
                Nsim=self.Nsim,
                SourcePhotons_template=gps_template,
                output_mapping=output_mapping
            )

    def process_results(self, EDP):
        SIF_simulation = np.zeros((self.Nupper, self.num_pixels, self.num_pixels))
        for i in range(self.Nupper):
            filename = f"beam{i}.txt"
            fpath = os.path.join(self.subfolder_name, filename)
            analyzer = EDP.Edep_analyser(fpath, self.num_pixels, self.num_pixels, detector_shape=self.tissue_box)
            SIF_simulation[i] = analyzer.analyse()
        self.SIF_simulation = SIF_simulation
        #return SIF_simulation

    def expand_SIF(self, BDE):
        self.SIF_total = BDE.expand_to_4Nbeams(self.SIF_simulation)
        #return self.SIF_total

    def save_SIF(self):
        os.makedirs(self.SIF_dir, exist_ok=True)
        np.save(self.SIF_filepath, self.SIF_total)
        print(f"SIF saved to: {self.SIF_filepath}")

    def process_data_and_save_SIF(self, EDP, BDE):
        self.process_results(EDP)
        self.expand_SIF(BDE)
        self.save_SIF()