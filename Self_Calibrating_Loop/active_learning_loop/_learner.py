import numpy as np
from scipy.integrate import odeint
import sys
import os
import copy
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from GCAD.get_system_equations_pop import system_equations_DsRed_pop
    from GCAD.define_circuit import Topo
except ImportError:
    print("Warning: Could not import system_equations.")

class Learner:
    """
    The Inference Engine (Multi-Circuit).
    Evaluates parameter models against MULTIPLE experiments simultaneously.
    """
    def __init__(self, circuit_dict, selection_ratio=0.2, perturbation_scale=0.15):
        self.circuit_dict = circuit_dict
        self.selection_ratio = selection_ratio
        self.perturbation_scale = perturbation_scale
        
        self.active_parts = set()
        for topology in self.circuit_dict.values():
            self.active_parts.update(topology.part_list)
        print(f"[Learner Init] Tracking and optimizing only active parts: {self.active_parts}")

    def update_belief(self, current_ensemble, promo_params, lab_data_dict, t_span):
        """
        Args:
            current_ensemble: List of universal parameter dictionaries.
            promo_params: The fixed promoter parameters.
            lab_data_dict: {(circuit_name, dose): y_noisy_array}
            t_span: Time array for simulation.
        """
        print(f"\n--- [Learner] Assessing {len(current_ensemble)} models against {len(lab_data_dict)} experiments ---")

        errors = []
        
        # 1. EVALUATE ENSEMBLE (Calculate Total Error)
        for p_mut in tqdm(current_ensemble, desc="Evaluating Models", unit="model"):
            total_nmse = 0.0
            
            # For this specific model, simulate EVERY experiment in the lab data
            for (c_name, dose), lab_y in lab_data_dict.items():
                topology = self.circuit_dict[c_name]
                
                # Setup temp topology for this dosage
                sim_dose = {k: v * dose for k, v in topology.dose.items() if k != 'Rep'}
                if 'Rep' in topology.dose: sim_dose['Rep'] = topology.dose['Rep']
                temp_topo = Topo(topology.edge_list, sim_dose, topology.promo_node)

                try:
                    y_raw = odeint(
                        system_equations_DsRed_pop,
                        np.zeros(temp_topo.num_states * 2),
                        t_span,
                        args=('on', np.ones(5), temp_topo, promo_params, p_mut)
                    )[:, -1]
                    
                    # Scale the prediction to match Lab data
                    y_pred = y_raw / 66.60528765956212
                    
                    # Prevent flatline explosion
                    data_scale = max(np.max(lab_y), 1.0)
                    nmse = np.mean(((y_pred - lab_y) / data_scale) ** 2)
                    
                    total_nmse += nmse
                    
                except Exception as e:
                    total_nmse += float('inf')
                    
            errors.append(total_nmse)

        errors = np.array(errors)

        # 2. SELECTION (Truncation Selection)
        cutoff_count = max(int(len(current_ensemble) * self.selection_ratio), 5) 
        best_indices = np.argsort(errors)[:cutoff_count]
        
        survivors = [current_ensemble[i] for i in best_indices]
        best_error = errors[best_indices[0]]
        
        print(f"[Learner] Selected top {len(survivors)} candidates.")
        print(f"[Learner] Best Total NMSE: {best_error:.4e} (across {len(lab_data_dict)} exps)")
        
        # 3. RESAMPLING (Perturbation)
        new_ensemble = []
        target_size = len(current_ensemble)
        
        while len(new_ensemble) < target_size:
            parent_idx = np.random.randint(len(survivors))
            child = copy.deepcopy(survivors[parent_idx])
            self._perturb_parameters(child)
            new_ensemble.append(child)
            
        return new_ensemble, best_error

    def _perturb_parameters(self, param_dict):
        """Helper: Applies multiplicative noise ONLY to targeted indices."""
        
        targets = {
            'Z6': [0, 1],
            'I13': [0] 
        }
        
        for part, indices in targets.items():
            if part in param_dict:
                for idx in indices:
                    # Only perturb the specific parameters we care about
                    noise_factor = np.random.normal(1.0, self.perturbation_scale)
                    param_dict[part][idx] *= noise_factor