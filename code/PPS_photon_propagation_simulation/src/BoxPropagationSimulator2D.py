import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt


class box_propagation_simulator2D:
    """
    Inherited from [link](https://github.com/MateHudo/PhotonPropagationMCSimulation/tree/master); 
    adjusted to 2D, going from box -> rectangle.
    Simulates photon propagation in a 2D rectangular medium, tracking each step,
    considers photoeffect (PE), Compton scattering (CS), and pair production (PP). Assumes local
    electron energy deposition and not braking radiation and no fluorescence photons (good aproximation in
    tissue, all-in-all those photons would be absorbed locally). 
    Data for attenuation coefficients is gotten from NIST XCOM, using soft tissue mixture.
    todo: incorporate NIST raw data and use LACloader method to calculate LAC from MAC!
    """
    def __init__(self, box_dimensions, LAC_data, Emin_terminate=0.001):
        # box dimensions (length and width)
        self.box_dimensions = box_dimensions
        self.box_length = box_dimensions[0]
        self.box_width = box_dimensions[1]
        self.define_box_related_parameters()
        
        # Define constants once at initialization instead of in methods
        self.Emin_terminate = Emin_terminate
        self.m_e = 0.511  # Electron rest mass energy in MeV
        self.pair_production_threshold = 1.022  # MeV
        self.annihilation_energy = 0.511  # MeV per gamma in pair production
        
        # LAC data
        self.LAC_data = LAC_data
        self.energy_list = self.LAC_data['energy']
        self.lac_cs_list = self.LAC_data['lac_cs']
        self.lac_pe_list = self.LAC_data['lac_pe'] 
        self.lac_pp_list = self.LAC_data['lac_pp']
        

    def define_box_related_parameters(self):
        nF = np.array([1, 0, 0])  # normal vector of front face
        nB = np.array([-1, 0, 0])  # normal vector of back face
        nL = np.array([0, 1, 0])  # normal vector of left face
        nR = np.array([0, -1, 0])  # normal vector of right face
        self.face_normals = np.array([nF, nB, nL, nR])

        rF = np.array([self.box_length / 2, 0, 0])  # Front face position
        rB = np.array([-self.box_length / 2, 0, 0]) # Back face position
        rL = np.array([0, self.box_width / 2, 0])  # Left face position
        rR = np.array([0, -self.box_width / 2, 0])  # Right face position
        self.r_faces = [rF, rB, rL, rR]  # List of points on the cuboid faces

    def NewDirectionUnitVector(self, u_old, theta_sc):
        u_old = u_old / np.linalg.norm(u_old)
        C, S = np.cos(theta_sc), np.sin(theta_sc)
        ux = C* u_old[0] - S * u_old[1]
        uy = S * u_old[0] + C * u_old[1]
        uz = 0
        return np.array([ux, uy, uz])  #! 2D case, z-component is zero

    def ComptonScatteringInteraction(self, E, u_particle):
        # Use pre-defined constant instead of defining it each time
        alpha = E / self.m_e
        eps_min = 1 / (1 + 2 * alpha)
        beta1 = - np.log(eps_min)
        beta2 = 0.5 * (1 - eps_min ** 2)
        while True:
            r1, r2, r3 = np.random.uniform(0, 1, 3)
            if r1 < beta1 / (beta1 + beta2):
                eps_prop = eps_min ** r2
            else:
                eps_prop = np.sqrt(eps_min ** 2 + (1 - eps_min ** 2) * r2)
            t = (1 - eps_prop) / alpha / eps_prop
            g_eps = 1 - eps_prop * t * (2 - t) / (1 + eps_prop ** 2)
            if g_eps > r3:
                eps = eps_prop
                break
        E_scattered = E * eps
        # in 2D, we can simplify the scattering direction, only theta angle, add possibility switching it to -theta  
        theta_cs = np.random.choice([-1,1]) * np.arccos(1 - 1 / alpha * (1 / eps - 1)) # theta = [-pi,pi]
        u_new = self.NewDirectionUnitVector(u_particle, theta_cs)
        return E_scattered, u_new

    def PhotonFreePath(self, E):
        # Use pre-extracted data instead of accessing dictionary each time
        lac_cs = np.interp(E, self.energy_list, self.lac_cs_list)
        lac_pe = np.interp(E, self.energy_list, self.lac_pe_list)
        u1, u2, u3 = np.random.uniform(0, 1, 3)
        s_cs = - np.log(u1) / lac_cs
        s_pe = - np.log(u2) / lac_pe
        s_pp = np.inf
        if E > self.pair_production_threshold:  # Use pre-defined constant
            lac_pp = np.interp(E, self.energy_list, self.lac_pp_list)
            s_pp = - np.log(u3) / lac_pp
        free_path_length = min(s_cs, s_pe, s_pp)
        if free_path_length == s_cs:
            interaction_type = 'compton'
        elif free_path_length == s_pe:
            interaction_type = 'phot'
        else:
            interaction_type = 'pair'
        return free_path_length, interaction_type

    def BoxPathLength2D(self, r_particle, u_particle):
        min_path_length = np.inf
        exit_plane_index = -1
        # seemingly only change is the number of planes - simply ignore planes up and down (3rd dimension)
        for plane_index in range(4):
            r_plane, n_plane = self.r_faces[plane_index], self.face_normals[plane_index]
            relative_direction_scaling_factor = np.dot(u_particle, n_plane)
            if relative_direction_scaling_factor <= 0:
                continue
            # this is a 3D calculation, we simply put z=0
            path_length_to_plane = np.abs(np.dot(n_plane, r_particle - r_plane)) / relative_direction_scaling_factor
            if path_length_to_plane < min_path_length:
                min_path_length = path_length_to_plane
                exit_plane_index = plane_index
        return min_path_length, exit_plane_index

    def ParticleStep(self, PreStepParticleInfo):
        #* need to tell where particle is, where it goes ant what is its energy!!
        r_particle = PreStepParticleInfo['r']
        u_particle = PreStepParticleInfo['u']
        E = PreStepParticleInfo['E']

        add_photons_to_simulate = [] 
        box_path_length, exit_plane_index = self.BoxPathLength2D(r_particle, u_particle)
        free_path_length, interaction_type = self.PhotonFreePath(E)
        
        if free_path_length > box_path_length:
            interaction_type = 'exit' # transportation
            E_dep = 0.0
            E_new = E
            u_new = u_particle
            r_new = r_particle + box_path_length * u_particle
        else:
            r_new = r_particle + free_path_length * u_particle
            exit_plane_index = -1
            if interaction_type == 'compton':
                E_new, u_new = self.ComptonScatteringInteraction(E, u_particle)
                E_dep = E - E_new
                if E_new > self.Emin_terminate:
                    add_photons_to_simulate.append({
                        'E': E_new,
                        'u': u_new,
                        'r': r_new,
                    })
            elif interaction_type == 'phot':
                E_new = 0.0
                E_dep = E
                u_new = None
            elif interaction_type == 'pair':
                E_new = 0.0
                E_dep = E - self.pair_production_threshold  # Use pre-defined constant
                u_new = None
                th = np.arccos(np.random.uniform(-1, 1))
                x, y, z = np.cos(th), np.sin(th), 0
                u_pp = np.array([x, y, z])
                add_photons_to_simulate.extend([
                    {'E': self.annihilation_energy, 'u': u_pp, 'r': r_new},  # Use pre-defined constant
                    {'E': self.annihilation_energy, 'u': -u_pp, 'r': r_new}  # Use pre-defined constant
                ])
            else:
                raise RuntimeError("Unknown interaction type.")
            
        return {
            'interaction': interaction_type,
            'E_dep': E_dep,
            'add_photons_to_simulate': add_photons_to_simulate,
            'E_new': E_new,
            'u_new': u_new,
            'r_new': r_new,
        }

    def rtg_beam_energy_spectrum(self, Ephotons_max, alpha, beta,doPlot=False):
        """ 
        Returns discrete Probability Density Function (PDF) of RTG Photon Energy Spectrum Model [rtgESM] 
        """
        #* hundred points is enought for now, Nsim must be much larger!!
        Ephotons = np.linspace(0.01, Ephotons_max, 100) 
        #$ rtgESM - effective model, simplified, alpha and beta shall be adjusted for each tube voltage!
        spectrum_model_function = Ephotons**alpha * np.exp(-beta * Ephotons)
        # discrete PDF normalization
        energy_pdf = spectrum_model_function / np.sum(spectrum_model_function)
        
        if doPlot:
            plt.figure(figsize=(8, 6))
            plt.plot(Ephotons, energy_pdf, label=f'RTG Spectrum Model (alpha={alpha}, beta={beta})')
            plt.xlabel('Photon Energy (MeV)')
            plt.ylabel('Probability Density Function (PDF)')
            plt.title('RTG Photon Energy Spectrum')
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.show()
        
        return Ephotons, energy_pdf

    def simulate(self, Nsim, SourcePhotons_template, output_mapping={"on": False}):
        """ 
        Simulates the propagation of photons through a defined box geometry.
        Uses method ParticleStep as the main simulation engine.
        Args:
            - Nsim (int): Number of simulation events.
            - SourcePhotons_template (dict): Template for the source photons, containing 'r', 'u', and 'E'.
            Template could describe monoenergetic pencil-beam or RTG photon sources. Could be also 
            expanded to approximate RT fan-beam etc.
        Returns:
            - 
        """
        self.Ephotons_list = [] # for storing photon energies - for rtg spectrum checking
        source_spectrum_type = SourcePhotons_template['spectrum_type']
        beam_index = SourcePhotons_template['beam_index']
        # Check if the source spectrum type is valid
        if source_spectrum_type == 'mono':
            Ephotons = SourcePhotons_template['Emono']
            print(f"\nStarting simulation with monoenergetical photons: E={Ephotons}; {Nsim} events; beam index={beam_index}")
        elif source_spectrum_type == 'rtg':
            Ephotons_max = SourcePhotons_template['Emax']
            alpha = SourcePhotons_template['alpha']
            beta = SourcePhotons_template['beta']   
            Ephotons,energy_pdf = self.rtg_beam_energy_spectrum(Ephotons_max, alpha, beta)
            print(f"\nStarting simulation with RTG photons: U={Ephotons_max}, MV; Nsim={Nsim}\n") 

        #* create a timestamped filename for saving simulation info
        timestamp_str = dt.now().strftime("%Y%m%d_%H%M%S")
        #* save the output data to initial_output folder
        output_file = f"./initial_output/run_{timestamp_str}.txt"
        #* if output mapping is provided, use it - create file directly in simulation_data folder
        if output_mapping["on"] == True:
            folder_name, file_name = output_mapping["folder_name"], output_mapping["file_name"]
            output_file = f"{folder_name}/{file_name}.txt"

        # define printing progress frequency
        progress_report_freq = 10  # %
        eid_progress_set = set(
            int(Nsim * pct / 100)
            for pct in range(progress_report_freq, 101, progress_report_freq)
        )

        #* loop over all simulation events
        for eid in range(Nsim):
            # Print progress every {progress_report_freq}% of Nsim
            if eid in eid_progress_set:
                print(f"\tProgress: {eid / Nsim * 100:.2f}%")

            #* initialize list of photons to simulate with new primary
            photons_to_simulate = [{
                'eid': eid,
                #'tid': 0,
                'r': SourcePhotons_template['r'],
                'u': SourcePhotons_template['u'],
            }]
            if source_spectrum_type == 'mono':
                photons_to_simulate[0]['E'] = Ephotons
            elif source_spectrum_type == 'rtg':
                #* Sample energy from the PDF
                Ertg_photon = np.random.choice(Ephotons, p=energy_pdf)  
                photons_to_simulate[0]['E'] = Ertg_photon
                self.Ephotons_list.append(Ertg_photon)
    
            #$ loop over all photons to simulate until none are left (photons_to_simulate is empty)
            while photons_to_simulate:
                # Process the first photon in the list
                PreStepData = photons_to_simulate.pop(0)
                # simulate the particle step
                result = self.ParticleStep(PreStepData)
                # extend the list of photons to simulate with new photons generated in this step
                photons_to_simulate.extend(result['add_photons_to_simulate'])
                
                # write the result to the file
                with open(output_file, "a") as f:
                    interaction_type = result['interaction']
                    Edep = result['E_dep']
                    rint = result['r_new']
                    xint, yint = rint[0], rint[1]  # 2D case, z is always 0
                    
                    #! write only if energy is deposited - aim for dose distibution
                    if Edep > 0:
                        f.write(
                            #f"{interaction_type} {Edep} {xint} {yint} \n"
                            f"{eid} {interaction_type} {Edep} {xint} {yint} \n"
                        )

        print("------- Simulation finished -------")