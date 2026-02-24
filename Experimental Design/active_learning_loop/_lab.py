import numpy as np
from scipy.integrate import odeint
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from circuits_miner.get_system_equations_pop import system_equations_DsRed_pop
except ImportError:
    print("Warning: Could not import system_equations.")

class VirtualLab:
    """
    The Ground Truth. 
    Holds the "TRUE" (Hidden) parameters and generates noisy data.
    """
    def __init__(self, topology, true_params, true_promo_params, ref_val=None, noise_level=0.05):
        """
        Args:
            topology (Topo): The circuit topology object.
            true_params (list): The 'true' parameters (hidden from the loop).
            true_promo_params (list): The promoter parameters.
            ref_val (float): Optional normalization factor (reference value).
            noise_level (float): Std dev of noise (relative to signal magnitude).
        """
        self.topology = topology
        self.true_params = true_params
        self.true_promo_params = true_promo_params
        self.ref_val = ref_val if ref_val is not None else 1.0
        self.noise_level = noise_level
        
    def run_experiment(self, dosage_factor, t_span=None):
        """
        The Interface. The AL-loop asks for a specific dosage, and gets back data.
        
        Args:
            dosage_factor (float): The scaling factor for the inputs (0.1x to 4.0x).
            t_span (np.array): Time points to measure. Default: 0-126h.
            
        Returns:
            np.array: Noisy fluorescence trace (Relative Units).
        """
        if t_span is None:
            t_span = np.arange(0, 126, 1)
            
        experimental_dose = {k: v * dosage_factor for k, v in self.topology.dose.items() if k != 'Rep'}
        if 'Rep' in self.topology.dose:
            experimental_dose['Rep'] = self.topology.dose['Rep']
        
        # Create a temporary topology with the new dosage
        from circuits_miner.define_circuit import Topo 
        lab_bench_topology = Topo(
            self.topology.edge_list, 
            experimental_dose, 
            self.topology.promo_node
        )
        
        y_raw = odeint(
            system_equations_DsRed_pop,
            np.zeros(lab_bench_topology.num_states * 2),
            t_span,
            args=('on', np.ones(5), lab_bench_topology, self.true_promo_params, self.true_params)
        )[:, -1]
        
        
        y_measured = y_raw / (self.ref_val + 1e-9)
        # Add Noise (Sensor Error)
        # Noise scales slightly with signal magnitude (heteroscedastic) + baseline noise
        noise = np.random.normal(0, self.noise_level * np.max(y_measured), size=len(y_measured))
        y_noisy = np.maximum(y_measured + noise, 0)
        
        return t_span, y_noisy