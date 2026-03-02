import numpy as np
from scipy.integrate import odeint
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from GCAD.get_system_equations_pop import system_equations_DsRed_pop
    from GCAD.define_circuit import Topo
except ImportError:
    print("Warning: Could not import system_equations or Topo.")

class VirtualLab:
    """
    The Ground Truth Multi-Circuit Lab.
    Holds the UNIVERSAL 'true' parameters and generates noisy data for specific circuits.
    """
    def __init__(self, circuit_dict, true_universal_params, true_promo_params, ref_val=66.60528765956212, noise_level=0.05):
        self.circuits = circuit_dict
        self.true_universal_params = true_universal_params
        self.true_promo_params = true_promo_params
        self.ref_val = ref_val if ref_val is not None else 1.0
        self.noise_level = noise_level

    def run_experiment(self, circuit_id, dosage_factor, t_span=None):
        if circuit_id not in self.circuits:
            raise ValueError(f"Circuit {circuit_id} not found in the lab.")
            
        topology = self.circuits[circuit_id]
        
        if t_span is None:
            t_span = np.arange(0, 126, 1)
            
        experimental_dose = {k: v * dosage_factor for k, v in topology.dose.items() if k != 'Rep'}
        if 'Rep' in topology.dose:
            experimental_dose['Rep'] = topology.dose['Rep']
        
        # 1. Create a temporary topology with the new dosage
        lab_bench_topology = Topo(
            topology.edge_list, 
            experimental_dose, 
            topology.promo_node
        )
        
        # 2. Run the ODE
        y_raw = odeint(
            system_equations_DsRed_pop,
            np.zeros(lab_bench_topology.num_states * 2),
            t_span,
            args=('on', np.ones(5), lab_bench_topology, self.true_promo_params, self.true_universal_params)
        )[:, -1]
        
        y_measured = y_raw / (self.ref_val + 1e-9)
        
        # 3. Add Noise (Relying on global notebook seed for reproducibility)
        noise = np.random.normal(0, self.noise_level * np.max(y_measured), size=len(y_measured))
        y_noisy = np.maximum(y_measured + noise, 0)
        
        return t_span, y_noisy