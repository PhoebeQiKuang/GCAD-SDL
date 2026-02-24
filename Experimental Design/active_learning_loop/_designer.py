import numpy as np
from scipy.integrate import odeint
import sys
import os

# Import physics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from circuits_miner.get_system_equations_pop import system_equations_DsRed_pop
    from circuits_miner.define_circuit import Topo
except ImportError:
    print("Warning: Could not import modules. Check your paths.")

class ExperimentDesigner:
    """
    Plans the next experiment by simulating all candidate dosages across the ensemble 
    of parameters and finding the most informative experiment.
    """
    # --> ADDED: variance_metric handle ('hybrid' or 'time_specific')
    def __init__(self, topology, candidate_dosages=None, variance_metric="time_specific"):
        self.topology = topology
        self.variance_metric = variance_metric
        if candidate_dosages is None:
            self.candidate_dosages = np.linspace(0.2, 4.0, 20) 
        else:
            self.candidate_dosages = candidate_dosages

    def design_experiment(self, parameter_ensemble, promo_params_ensemble):
        """
        The OED Logic.
        1. Simulates ALL dosages.
        2. Calculates variance based on the chosen metric.
        3. Normalizes and scores dosages.
        """
        t_span = np.arange(0, 126, 1)
        raw_scores = [] 
        
        # --- PHASE 1: SIMULATE ALL DOSAGES ---
        for dose in self.candidate_dosages:
            
            # Setup Temp Topology
            current_dose_map = {k: v * dose for k, v in self.topology.dose.items() if k != 'Rep'}
            if 'Rep' in self.topology.dose: current_dose_map['Rep'] = self.topology.dose['Rep']
            temp_topo = Topo(self.topology.edge_list, current_dose_map, self.topology.promo_node)
            
            # Simulate Ensemble
            traces = []
            for p_mut, p_promo in zip(parameter_ensemble, promo_params_ensemble):
                y = odeint(
                    system_equations_DsRed_pop,
                    np.zeros(temp_topo.num_states * 2),
                    t_span,
                    args=('on', np.ones(5), temp_topo, p_promo, p_mut)
                )[:, -1]
                traces.append(y)
            
            traces = np.array(traces) # Shape: (200 models, 126 timepoints)
            
            # --- PHASE 2: CALCULATE VARIANCE METRIC ---
            if self.variance_metric == "time_specific":
                # NEW METRIC: Raw Dynamics V(t)
                v_t = np.var(traces, axis=0)      # Variance across columns
                mean_hourly_var = np.mean(v_t)    # Compress to single score
                raw_scores.append(mean_hourly_var)
                
            elif self.variance_metric == "hybrid":
                # OLD METRIC: Feature Extraction
                vp, vt = self._get_raw_variances(traces)
                raw_scores.append((vp, vt)) # Store as tuple temporarily
                
            else:
                raise ValueError("variance_metric must be 'hybrid' or 'time_specific'")
            
        # --- PHASE 3: NORMALIZE & AGGREGATE ---
        if self.variance_metric == "time_specific":
            raw_scores = np.array(raw_scores)
            max_uncertainty = np.max(raw_scores) # Extract highest absolute variance for Convergence Handle
            info_scores = raw_scores / (max_uncertainty + 1e-9)
            
        elif self.variance_metric == "hybrid":
            raw_var_p = np.array([x[0] for x in raw_scores])
            raw_var_t = np.array([x[1] for x in raw_scores])
            
            max_uncertainty = np.max(raw_var_p) # Use prominence variance for Convergence Handle
            norm_p = raw_var_p / (max_uncertainty + 1e-9)
            norm_t = raw_var_t / (np.max(raw_var_t) + 1e-9)
            
            info_scores = 0.7 * norm_p + 0.3 * norm_t

        # --- PHASE 4: DECISION ---
        best_idx = np.argmax(info_scores)
        best_dose = self.candidate_dosages[best_idx]
        
        # --> ADDED: Return max_uncertainty as our "Stop Handle"
        return best_dose, info_scores, self.candidate_dosages, max_uncertainty

    def _get_raw_variances(self, traces):
        """Helper: Extracts raw features for Hybrid Metric."""
        proms = []
        times = []
        for trace in traces:
            peak_idx = np.argmax(trace)
            peak_val = trace[peak_idx]
            final_val = trace[-1]
            
            relative_drop = (peak_val - final_val) / (peak_val + 1e-9)
            
            if relative_drop < 0.2 or peak_idx == 0 or peak_idx == len(trace)-1:
                proms.append(0.0) # Failure (Unconditional)
            else:
                proms.append(peak_val - final_val)
                times.append(peak_idx)
        
        var_p = np.var(proms)
        var_t = np.var(times) if len(times) > 5 else 0.0
            
        return var_p, var_t