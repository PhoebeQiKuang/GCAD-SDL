import numpy as np
from scipy.integrate import odeint
import sys
import os
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from circuits_miner.get_system_equations_pop import system_equations_DsRed_pop
    from circuits_miner.define_circuit import Topo
except ImportError:
    print("Warning: Could not import system_equations.")

class Learner:
    """
    The Inference Engine.
    
    Implements an Approximate Bayesian Computation (ABC-SMC) approach to update 
    the parameter ensemble based on experimental data.
    
    Process:
    1. Simulation: Run the current ensemble under experimental conditions.
    2. Assessment: Calculate the error (MSE) between simulation and Lab data.
    3. Selection: Retain high-likelihood parameters (posterior approximation).
    4. Resampling: Perturb selected parameters to maintain ensemble diversity.
    """
    def __init__(self, topology, selection_ratio=0.2, perturbation_scale=0.1):
        """
        Args:
            topology (Topo): The circuit topology structure.
            selection_ratio (float): Fraction of the ensemble to retain (Top N%).
            perturbation_scale (float): Standard deviation for parameter perturbation (relative).
        """
        self.topology = topology
        self.selection_ratio = selection_ratio
        self.perturbation_scale = perturbation_scale

    def update_belief(self, current_ensemble, current_promos, experiment_dose, lab_data_t, lab_data_y):
        """
        Updates the parameter ensemble given new experimental data.
        
        Args:
            current_ensemble (list): List of parameter dictionaries (Prior).
            current_promos (list): List of corresponding promoter parameters.
            experiment_dose (float): The dosage factor used in the experiment.
            lab_data_t (np.array): Time points from the experiment.
            lab_data_y (np.array): Normalized fluorescence data from the experiment.
            
        Returns:
            tuple: (updated_ensemble, updated_promos, best_error) represents the Posterior.
        """
        print(f"   [Learner] Assessing {len(current_ensemble)} parameter sets...")

        # 1. CONFIGURE SIMULATION
        # Create a temporary topology reflecting the experimental dosage
        sim_dose = {k: v * experiment_dose for k, v in self.topology.dose.items() if k != 'Rep'}
        if 'Rep' in self.topology.dose: sim_dose['Rep'] = self.topology.dose['Rep']
        temp_topo = Topo(self.topology.edge_list, sim_dose, self.topology.promo_node)
        
        # 2. EVALUATE ENSEMBLE (Calculate Error)
        errors = []
        for i, (p_mut, p_promo) in enumerate(zip(current_ensemble, current_promos)):
            try:
                y_pred = odeint(
                    system_equations_DsRed_pop,
                    np.zeros(temp_topo.num_states * 2),
                    lab_data_t,
                    args=('on', np.ones(5), temp_topo, p_promo, p_mut)
                )[:, -1]
                
                data_range = np.max(lab_data_y) - np.min(lab_data_y) + 1e-9
                mse = np.mean(((y_pred - lab_data_y) / data_range) ** 2)
                errors.append(mse)
            except Exception as e:
                errors.append(float('inf'))

        errors = np.array(errors)

        # 3. SELECTION (Truncation Selection)
        # Identify the indices of the best-fitting parameters
        cutoff_count = max(int(len(current_ensemble) * self.selection_ratio), 5) 
        best_indices = np.argsort(errors)[:cutoff_count]
        
        survivors = [current_ensemble[i] for i in best_indices]
        survivor_promos = [current_promos[i] for i in best_indices]
        best_error = errors[best_indices[0]]
        
        print(f"   [Learner] Selected top {len(survivors)} candidates. Min Error (NMSE): {best_error:.4e}")
        
        # 4. RESAMPLING (Kernel Density Estimation / Perturbation)
        # Regenerate the population to original size by perturbing survivors
        new_ensemble = []
        new_promos = []
        target_size = len(current_ensemble)
        
        while len(new_ensemble) < target_size:
            # Randomly select a parent from the survivors
            # Create a perturbed child
            parent_idx = np.random.randint(len(survivors))

            child = copy.deepcopy(survivors[parent_idx])
            self._perturb_parameters(child)

            new_ensemble.append(child)
            new_promos.append(survivor_promos[parent_idx]) # Promoter params assumed fixed/linked for now
            
        # --> ADDED: Returning best_error so main_loop can monitor fitting quality
        return new_ensemble, new_promos, best_error

    def _perturb_parameters(self, param_dict):
        """Helper: Applies multiplicative Gaussian noise."""
        for key in param_dict:
            noise_factor = np.random.normal(1.0, self.perturbation_scale)
            # Handle list parameters 
            # We perturb each element independently
            if isinstance(param_dict[key], list):
                param_dict[key] = [v * np.random.normal(1.0, self.perturbation_scale) for v in param_dict[key]]
            # Handle scalar parameters
            elif isinstance(param_dict[key], (float, int)):
                param_dict[key] *= noise_factor