import numpy as np
from scipy.integrate import odeint
import sys
import os
from tqdm import tqdm

# Import physics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from GCAD.get_system_equations_pop import system_equations_DsRed_pop
    from GCAD.define_circuit import Topo
except ImportError:
    print("Warning: Could not import modules. Check your paths.")

class ExperimentDesigner:
    def __init__(self, circuit_dict, candidate_dosages=None, variance_metric="time_specific", p_circuits=2, q_dosages=2):
        self.circuit_dict = circuit_dict
        self.variance_metric = variance_metric
        self.p_circuits = min(p_circuits, len(circuit_dict))
        
        if candidate_dosages is None:
            self.candidate_dosages = np.arange(0.2, 4.2, 0.2)
        else:
            self.candidate_dosages = candidate_dosages
            
        self.q_dosages = min(q_dosages, len(self.candidate_dosages))

    def design_experiment(self, belief_cloud, promo_params):
        t_span = np.arange(0, 126, 1)
        
        circuit_names = list(self.circuit_dict.keys())
        num_circuits = len(circuit_names)
        num_dosages = len(self.candidate_dosages)
        
        variance_matrix = np.zeros((num_circuits, num_dosages))
        raw_vp_matrix = np.zeros((num_circuits, num_dosages))
        raw_vt_matrix = np.zeros((num_circuits, num_dosages))
        
        all_simulations = {} 
        
        total_tasks = num_circuits * num_dosages
        with tqdm(total=total_tasks, desc="Simulating 2D Grid", unit="exp") as pbar:
            
            for i, c_name in enumerate(circuit_names):
                topology = self.circuit_dict[c_name]
                
                for j, dose in enumerate(self.candidate_dosages):
                    current_dose_map = {k: v * dose for k, v in topology.dose.items() if k != 'Rep'}
                    if 'Rep' in topology.dose: current_dose_map['Rep'] = topology.dose['Rep']
                    temp_topo = Topo(topology.edge_list, current_dose_map, topology.promo_node)

                    traces = []
                    for param_dict in belief_cloud:
                        y_raw = odeint(
                            system_equations_DsRed_pop,
                            np.zeros(temp_topo.num_states * 2),
                            t_span,
                            args=('on', np.ones(5), temp_topo, promo_params, param_dict)
                        )[:, -1]
                        
                        # FIX: Scale Designer traces
                        y_scaled = y_raw / 66.60528765956212
                        traces.append(y_scaled)
                    
                    all_simulations[(c_name, dose)] = traces 
                    
                    if self.variance_metric == "time_specific":
                        v_t = np.var(traces, axis=0)
                        variance_matrix[i, j] = np.mean(v_t)
                        
                    elif self.variance_metric == "hybrid":
                        vp, vt = self._get_raw_variances(traces)
                        raw_vp_matrix[i, j] = vp
                        raw_vt_matrix[i, j] = vt
                        
                    pbar.update(1) 

        if self.variance_metric == "hybrid":
            max_p = np.max(raw_vp_matrix) + 1e-9
            max_t = np.max(raw_vt_matrix) + 1e-9
            variance_matrix = (0.7 * (raw_vp_matrix / max_p)) + (0.3 * (raw_vt_matrix / max_t))

        selected_experiments, selected_matrix_indices = [], []
        row_totals = np.sum(variance_matrix, axis=1)
        top_p_circuit_indices = np.argsort(row_totals)[-self.p_circuits:][::-1]
        
        for c_idx in top_p_circuit_indices:
            c_name = circuit_names[c_idx]
            row_variances = variance_matrix[c_idx, :]
            top_q_dose_indices = np.argsort(row_variances)[-self.q_dosages:][::-1]
            for d_idx in top_q_dose_indices:
                dose = self.candidate_dosages[d_idx]
                selected_experiments.append((c_name, dose))
                selected_matrix_indices.append((c_idx, d_idx))
                
        max_uncertainty = np.max(variance_matrix)
        
        return selected_experiments, variance_matrix, selected_matrix_indices, max_uncertainty, all_simulations

    def _get_raw_variances(self, traces):
        proms, times = [], []
        for trace in traces:
            peak_idx = np.argmax(trace)
            peak_val = trace[peak_idx]
            final_val = trace[-1]
            relative_drop = (peak_val - final_val) / (peak_val + 1e-9)
            if relative_drop < 0.2 or peak_idx == 0 or peak_idx == len(trace)-1:
                proms.append(0.0)
            else:
                proms.append(peak_val - final_val)
                times.append(peak_idx)
        var_p = np.var(proms)
        var_t = np.var(times) if len(times) > 5 else 0.0
        return var_p, var_t