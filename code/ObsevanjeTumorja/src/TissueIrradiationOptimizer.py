import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.optimize import linprog

class tissue_irradiation_optimizer:
    def __init__(self, problem_parameters):
        self.problem_parameters = problem_parameters
        self.optimized = False

        # extract parameters from problem_parameters
        # fixme: this could be inherited from TissueIrradiationProblem class; explore!
        self.nrows = problem_parameters["nrows"]
        self.ncols = problem_parameters["ncols"]
        self.ncells = self.nrows * self.ncols
        self.nbeams = 2 * (self.nrows + self.ncols)
        self.tissue_matrix = problem_parameters["tissue_matrix"]
        self.c_beams = problem_parameters["c_beams"]
        self.A_tissue = problem_parameters.get("A_tissue", None)
        self.A_tumor = problem_parameters.get("A_tumor", None)
        self.A_healthy = problem_parameters.get("A_healthy", None)
        self.A_ub = problem_parameters["A_ub"]
        self.b_ub = problem_parameters["b_ub"]
        self.A_eq = problem_parameters.get("A_eq", None)
        self.b_eq = problem_parameters.get("b_eq", None)
        self.beam_type = problem_parameters["beam_type"]
        self.lac = problem_parameters.get("lac", 1.0)

        #self.optimal_beam_intensities = None
        self.optimal_beam_intensities = np.zeros(self.nbeams)  # Initialize with zeros
        self.DH_tot = -99

    def optimize(self, doReport=False):        
        self.result = linprog(c=self.c_beams, A_ub=self.A_ub, b_ub=self.b_ub, method='highs',A_eq=self.A_eq, b_eq=self.b_eq)

        if self.result.success:
            self.optimal_beam_intensities = self.result.x
            self.DH_tot = self.result.fun
            self.optimized = True
            if doReport:
                print("Optimization successful.")
                print("Optimal value:", self.result.fun)
                print("Optimal solution:", self.result.x)
            # calculate dose distribution
            self.calculate_dose_distribution()
        else:
            print("Optimization failed:", self.result.message)

    def calculate_dose_distribution(self):
        if not self.optimized:
            print("Run optimize() first.")
            return
        self.tissue_dose_matrix = self.A_tissue @ self.optimal_beam_intensities.reshape(-1, 1) # reshape: make a column out of line
        self.tissue_dose_matrix = self.tissue_dose_matrix.reshape(self.nrows, self.ncols) # reshape to tissue matrix shape

    def plot_dose_distribution(self):
        if not self.optimized:
            print("Run optimize() first.")
            return

        fig, ax = plt.subplots(figsize=(4, 4))
        cmap = colors.LinearSegmentedColormap.from_list('custom_bw', ['white', 'black'])
        im = ax.imshow(self.tissue_dose_matrix, cmap=cmap, interpolation='nearest', 
                       vmin=np.min(self.tissue_dose_matrix), vmax=np.max(self.tissue_dose_matrix),
                       extent=[-0.5, self.ncols-0.5, self.nrows-0.5, -0.5])

        # Draw grid lines for cells (only at cell boundaries)
        ax.set_xticks(np.arange(-0.5, self.ncols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.nrows, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_xlim(-0.5, self.ncols-0.5)
        ax.set_ylim(self.nrows-0.5, -0.5)

        # Annotate each cell with its dose (if nonzero)
        for i in range(self.nrows):
            for j in range(self.ncols):
                dose = self.tissue_dose_matrix[i, j]
                if dose > 0:
                    ax.text(j, i, f"{dose:.0f}", ha='center', va='center', color='red', fontsize=12, fontweight='bold')

        # Highlight tumor cells with red edges
        tumor_mask = (self.tissue_matrix == 1)
        for (i, j) in np.argwhere(tumor_mask):
            rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

        plt.title('Porazdelitev doze po tkivu')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()

    def plot_irradiation_scheme(self):
        if not self.optimized:
            print("Run optimize() first.")
            return

        fig, ax = plt.subplots(figsize=(8, 8))
        cmap = colors.LinearSegmentedColormap.from_list('custom_bw', ['white', 'black'])
        im = ax.imshow(self.tissue_dose_matrix, cmap=cmap, interpolation='nearest', 
                       vmin=np.min(self.tissue_dose_matrix), vmax=np.max(self.tissue_dose_matrix),
                       extent=[-0.5, self.ncols-0.5, self.nrows-0.5, -0.5])

        # Draw grid lines for cells (only at cell boundaries)
        ax.set_xticks(np.arange(-0.5, self.ncols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.nrows, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_xlim(-0.5, self.ncols-0.5)
        ax.set_ylim(self.nrows-0.5, -0.5)

        # Annotate each cell with its dose (if nonzero)
        for i in range(self.nrows):
            for j in range(self.ncols):
                dose = self.tissue_dose_matrix[i, j]
                if dose > 0:
                    ax.text(j, i, f"{dose:.0f}", ha='center', va='center', color='red', fontsize=12, fontweight='bold')

        # Highlight tumor cells with red edges
        tumor_mask = (self.tissue_matrix == 1)
        for (i, j) in np.argwhere(tumor_mask):
            rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, linewidth=4, edgecolor='darkred', facecolor='none')
            ax.add_patch(rect)

        # Highlight tissue edges
        ax.plot([-0.5, self.ncols-0.5], [-0.5, -0.5], color='black', linewidth=2)
        ax.plot([-0.5, self.ncols-0.5], [self.nrows-0.5, self.nrows-0.5], color='black', linewidth=2)
        ax.plot([-0.5, -0.5], [-0.5, self.nrows-0.5], color='black', linewidth=2)
        ax.plot([self.ncols-0.5, self.ncols-0.5], [-0.5, self.nrows-0.5], color='black', linewidth=2)
        
        # Draw beams: horizontal (rows) and vertical (cols) with smaller arrows at the edges
        arrow_length = 0.7
        arrow_head_width = 0.12
        arrow_head_length = 0.12
        ## beam intensities from left and right directions
        for beam_idx in range(self.nrows):
            intensity_left = self.optimal_beam_intensities[beam_idx]
            intensity_right = self.optimal_beam_intensities[self.nrows + beam_idx]
            if intensity_left > 0:
                # Left edge arrow for row beams
                ax.arrow(-0.9, beam_idx, arrow_length, 0, head_width=arrow_head_width, head_length=arrow_head_length,
                         fc='blue', ec='blue', linewidth=1.2, length_includes_head=True, zorder=10)
                ax.text(-0.9, beam_idx, f"{intensity_left:.0f}", va='center', ha='right', color='blue', fontsize=11)
            if intensity_right > 0:
                # Right edge arrow for row beams
                ax.arrow(self.ncols-0.0, beam_idx, -arrow_length, 0, head_width=arrow_head_width, head_length=arrow_head_length,
                         fc='blue', ec='blue', linewidth=1.2, length_includes_head=True, zorder=10)
                ax.text(self.ncols-0.3, beam_idx, f"{intensity_right:.0f}", va='center', ha='left', color='blue', fontsize=11)
        
        # intensities from top and bottom directions
        for beam_idx in range(self.ncols):
            intensity_top = self.optimal_beam_intensities[2*self.nrows + beam_idx]
            intensity_bottom = self.optimal_beam_intensities[2*self.nrows + self.ncols + beam_idx]
            if intensity_top > 0:
                # Top edge arrow for column beams
                ax.arrow(beam_idx, -0.9, 0, arrow_length, head_width=arrow_head_width, head_length=arrow_head_length,
                         fc='green', ec='green', linewidth=1.2, length_includes_head=True, zorder=10)
                ax.text(beam_idx, -0.7, f"{intensity_top:.0f}", va='bottom', ha='center', color='green', fontsize=11)
            if intensity_bottom > 0:
                # Bottom edge arrow for column beams
                ax.arrow(beam_idx, self.nrows-0.3, 0, -arrow_length, head_width=arrow_head_width, head_length=arrow_head_length,
                         fc='green', ec='green', linewidth=1.2, length_includes_head=True, zorder=10)
                ax.text(beam_idx, self.nrows-0.3, f"{intensity_bottom:.0f}", va='top', ha='center', color='green', fontsize=11)

        if self.beam_type == 'constant':
            beam_type_message = "konstantna intenziteta"
        elif self.beam_type == 'exponential':
            lambda_sign = 'λ'
            beam_type_message = f"eksponentna intenziteta ({lambda_sign}={1/self.lac} celica)"
        elif self.beam_type == 'simulated':
            beam_type_message = "simulirana intenziteta"
        plt.title(f'Obsevalni načrt in dozna porazdelitev\n{beam_type_message}\n\n',alpha=0.8)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()

    def report_statistics(self):
        print("Statistics:")
        if not hasattr(self, 'tissue_dose_matrix'): 
            print("Run calculate_dose_distribution() first.")
            return  
        tumor_mask = (self.tissue_matrix == 1)
        healthy_mask = (self.tissue_matrix == 0)
        avg_dose_tumor = self.tissue_dose_matrix[tumor_mask].mean() if np.any(tumor_mask) else 0
        avg_dose_healthy = self.tissue_dose_matrix[healthy_mask].mean() if np.any(healthy_mask) else 0
        print(f"Average dose to tumor cells: {avg_dose_tumor:.2f}")
        print(f"Average dose to healthy cells: {avg_dose_healthy:.2f}")
