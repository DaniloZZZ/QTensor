from NoiseChannels import *
from NoiseSimulator import * 
from NoiseModel import *
from NoiseSimComparisonResult import *
from helper_functions import *
import qiskit.providers.aer.noise as noise
from qtensor.tests.test_composers import *


### Noise model, simulation, and results of a QAOA algorithmm  ###
prob_1 = 0.01
prob_2 = 0.1

# Qiskit Noise Model
depol_chan_qiskit_1Q = noise.depolarizing_error(prob_1, 1)
depol_chan_qiskit_2Q = noise.depolarizing_error(prob_2, 2)

noise_model_qiskit = noise.NoiseModel()
noise_model_qiskit.add_all_qubit_quantum_error(depol_chan_qiskit_1Q, ['h', 'rx', 'rz'])
noise_model_qiskit.add_all_qubit_quantum_error(depol_chan_qiskit_2Q, ['cx'])

# QTensor Noise Model
depol_chan_qtensor_1Q = DepolarizingChannel(prob_1, 1)
depol_chan_qtensor_2Q = DepolarizingChannel(prob_2, 2)

noise_model_qtensor = NoiseModel()
noise_model_qtensor.add_channel_to_all_qubits(depol_chan_qtensor_1Q, ['H', 'XPhase', 'ZPhase'])
noise_model_qtensor.add_channel_to_all_qubits(depol_chan_qtensor_2Q, ['cX'])

min_n = 5
max_n = 7

min_p = 2
max_p = 3

min_d = 2
max_d = 3

num_samples = 3
num_circs_list = [10, 18, 32, 100, 178, 316]
num_nodes = 1
num_jobs_per_node = 3

outfile_name = '{}.json'.format(datetime.now().isoformat())
results = []

for n in range(min_n, max_n):
    for p in range(min_p, max_p):
        for d in range(min_d, max_d):
            # n * d must be even
            if (n * d) % 2 != 0:
                if d != max_d:
                    d += 1
                else:  
                    break
            if not 0 <= d < n:
                break

            for _ in range(num_samples):
                print("\n\nnum_circs_list:", num_circs_list)
                comparison = QAOAComparisonSimulator(n, p, d, noise_model_qiskit, noise_model_qtensor, num_circs_list)
                comparison.qtensor_qiskit_noisy_qaoa_mpi(num_nodes=num_nodes, num_jobs_per_node=num_jobs_per_node)
                results.extend(comparison.results)

save_dict_to_file(results, name = outfile_name) 