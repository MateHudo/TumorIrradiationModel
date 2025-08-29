import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

class tissue_irradiation_problem:
    def __init__(self, tissue_matrix): 
    #! Thing to consider - where to initialize variables: at class instance initializaton or at method begining? #######
        self.tissue_matrix = np.array(tissue_matrix)
        self.nrows, self.ncols = self.tissue_matrix.shape
        self.ncells = self.nrows * self.ncols
        self._analyze_tissue()
        
    def plot_tissue(self):
        print("Plotting tissue matrix...")
        cmap = colors.LinearSegmentedColormap.from_list('custom_bw', ['white', 'black'])
        plt.imshow(self.tissue_matrix, cmap=cmap, interpolation='nearest', vmin=0, vmax=1)
        plt.title('Tkivo')
        ax = plt.gca()
        # Set minor ticks at cell edges
        ax.set_xticks(np.arange(-0.5, self.ncols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.nrows, 1), minor=True)
        # Draw gridlines at minor ticks
        ax.grid(which='minor', color='red', linestyle='-', linewidth=1)
        # Make sure ticks are not shown
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        plt.show()

    def _analyze_tissue(self):
        self.tumor_positions = np.argwhere(self.tissue_matrix == 1)
        self.healthy_positions = np.argwhere(self.tissue_matrix == 0)
        self.n_tumor = len(self.tumor_positions)
        self.n_healthy = len(self.healthy_positions)
        self.n_beams = 2 * (self.nrows + self.ncols)

    def prepare_data_for_linprog(self,cell_dose_constr,beam_type='constant', lac=1.0,sim_irr_factors=None):
        """
        Prepare the data for linear programming optimization.
        This method creates the necessary matrices and vectors for the optimization problem.
        """
        self.beam_type = beam_type
        self.lac = lac
        self.simulation_irradiation_factors = sim_irr_factors
        self.cell_dose_constr = cell_dose_constr  
        self.min_dose_tumor = cell_dose_constr['min_dose_tumor']
        self.max_dose_healthy = cell_dose_constr.get('max_dose_healthy', None) # this one is optional
        
        #* run methods to prepare data for linear programming
        self.create_dose_distribution_matrix()
        self.define_linprog_inputs()
        print("Linear programming inputs defined.")

        return {
            "c_beams": self.c_beams,
            "A_ub": self.A_ub,
            "b_ub": self.b_ub,
            "A_eq": self.A_eq,
            "b_eq": self.b_eq,
            "A_tissue": self.A_tissue,
            "A_tumor": self.A_tumor,
            "A_healthy": self.A_healthy,
            "nrows": self.nrows,
            "ncols": self.ncols,
            "nbeams": self.n_beams,
            "ncells": self.ncells,
            "beam_type": self.beam_type,
            "tissue_matrix": self.tissue_matrix,
            "beam_type": self.beam_type,
            "lac": self.lac,
            # "simulation_irradiation_factors": self.simulation_irradiation_factors,
            # "min_dose_tumor": self.min_dose_tumor,
            # "max_dose_healthy": self.max_dose_healthy,
            # "delta_kij": self.delta_kij,
            # "n_tumor": self.n_tumor,
            # "n_healthy": self.n_healthy,
            # "tumor_positions": self.tumor_positions,
            # "healthy_positions": self.healthy_positions,

        }
    
    def create_dose_distribution_matrix(self):
        """
        Create the 3D dose distribution matrix (delta_kij) for the tissue matrix.
        Uses N_rows and N_cols from the notebook variables if available.
        Returns a numpy array of shape (N_beams, N_rows, N_cols).
        """
        # Use N_rows and N_cols from notebook if available, else from self
        nrows = self.nrows
        ncols = self.ncols
        n_beams = self.n_beams
        delta_kij = np.zeros((n_beams, nrows, ncols), dtype=float)

        # Fill delta_kij for all beams in a single loop, separating beam_type only once
        if self.beam_type == 'constant':
            print("Using constant intensity beam model.")
            # Left beams (rows)
            for i in range(nrows):
                for j in range(ncols):
                    delta_kij[i, i, j] = 1.0
            # Right beams (rows)
            for i in range(nrows):
                for j in range(ncols):
                    delta_kij[nrows + i, i, j] = 1.0
            # Top beams (cols)
            for j in range(ncols):
                for i in range(nrows):
                    delta_kij[2 * nrows + j, i, j] = 1.0
            # Bottom beams (cols)
            for j in range(ncols):
                for i in range(nrows):
                    delta_kij[2 * nrows + ncols + j, i, j] = 1.0
        
        elif self.beam_type == 'exponential':
            print("Using exponential intensity beam model with linear attenuation coefficient (lac):", self.lac, "/cm")
            lac = self.lac
            # Left beams (rows)
            for i in range(nrows):
                for j in range(ncols):
                    delta_kij[i, i, j] = np.exp(-lac * j)
            # Right beams (rows)
            for i in range(nrows):
                for j in range(ncols):
                    delta_kij[nrows + i, i, j] = np.exp(-lac * (ncols - j - 1))
            # Top beams (cols)
            for j in range(ncols):
                for i in range(nrows):
                    delta_kij[2 * nrows + j, i, j] = np.exp(-lac * i)
            # Bottom beams (cols)
            for j in range(ncols):
                for i in range(nrows):
                    delta_kij[2 * nrows + ncols + j, i, j] = np.exp(-lac * (nrows - i - 1))
        
        elif self.beam_type == 'simulated':
            print("Using beam from self simulated data for energy deposition distribution.")
            # Simulated beams logic would go here, similar to the above but using simulated data
            # For now, we will just fill with zeros as a placeholder
            if self.simulation_irradiation_factors is None:
                raise ValueError("Simulation_irradiation_factors must be provided for 'simulated' beam type.")
            delta_kij = self.simulation_irradiation_factors

        elif self.beam_type in ['proton', 'neutron', 'electron', 'other_stuff']:
            raise NotImplementedError(f"Beam type '{self.beam_type}' is not implemented yet.")
        else:
            raise ValueError(f"Unknown beam_type: {self.beam_type}")

        # at the end, make delta_kij factors matrix an attribute
        self.delta_kij = delta_kij

    def define_linprog_inputs(self):
        self.c_beams = np.zeros(self.n_beams, dtype=float)
        self.A_tissue = np.zeros((self.ncells, self.n_beams))
        self.A_tumor = []
        self.A_healthy = []
        self.A_eq = None
        self.b_eq = None

        # define c_beams
        for i in range(self.n_beams):
            self.c_beams[i] = np.sum(self.delta_kij[i, self.tissue_matrix == 0])  # Sum over healthy cells for constant beam
        # define A_tissue
        for idx in range(self.ncells):
            i = idx // self.ncols
            j = idx % self.ncols
            row = np.zeros(self.n_beams, dtype=float)
            for k in range(self.n_beams):
                row[k] = self.delta_kij[k, i, j]
            self.A_tissue[idx] = row
            if self.tissue_matrix[i, j] == 1:
                self.A_tumor.append(row)
            else:
                self.A_healthy.append(row)
        
        #* define A_eq and b_eq if constant beam type - because we cancel beams from the right and bottom
        if self.beam_type == 'constant':
            # add A_eq for dead beams (set right and bottom beams to zero)
            dead_beams_matrix = np.zeros((self.nrows + self.ncols, self.n_beams), dtype=int)
            for i in range(self.nrows):
                dead_beams_matrix[i, self.nrows + i] = 1  # right beams
            for j in range(self.ncols):
                dead_beams_matrix[self.nrows + j, 2 * self.nrows + self.ncols + j] = 1  # bottom beams
            self.A_eq = dead_beams_matrix
            self.b_eq = np.zeros(self.nrows + self.ncols)

        self.A_tumor = np.array(self.A_tumor)
        self.A_healthy = np.array(self.A_healthy)
        
        #* define A_upper_bound - just A_tumor and b_ub, but both with minus as we minimize!
        self.A_ub = -np.array(self.A_tumor)  # MINUS!!!
        self.b_ub = -np.ones(self.n_tumor) * self.min_dose_tumor  # MINUS!!!
        
        #* add A_healthy and extend b_ub if max dose to H cells is prescribed
        if self.max_dose_healthy is not None:
            self.A_ub = np.vstack((self.A_ub, self.A_healthy))
            self.b_ub = np.concatenate((self.b_ub, np.ones(self.n_healthy) * self.max_dose_healthy))
            
