from NoiseSimulator import NoiseSimulator
from NoiseModel import NoiseModel 
from NoiseSimComparisonResult import NoiseSimComparisonResult
from helper_functions import get_qaoa_params
from qtensor.Simulate import QtreeSimulator, NumpyBackend
from qtensor.contraction_backends import CuPyBackend
from qtensor import QiskitQAOAComposer
from qtree.operators import from_qiskit_circuit
from qtensor import tools

from qiskit import execute
from qiskit.providers.aer import AerSimulator, noise, AerError

import numpy as np
from sigpy import get_array_module
from qtensor.tools.lazy_import import cupy as cp
from itertools import repeat
import time

class ComparisonSimulator:
    def __init__(self):
        pass

class QAOAComparisonSimulator(ComparisonSimulator):
    def __init__(self, n: int, p: int, d: int, noise_model_qiskit: noise.NoiseModel, noise_model_qtensor: NoiseModel, num_circs_list: list):
        self.n = n
        self.p = p
        self.d = d
        self.noise_model_qiskit = noise_model_qiskit
        self.noise_model_qtensor = noise_model_qtensor
        self.num_circs_list = num_circs_list
        self.num_circs_simulated = []
        self.results = []
        self.recompute_previous_ensemble: bool

        self._check_params()
        self.num_circs_list.sort()

    def qtensor_qiskit_noisy_qaoa(self, backend = NumpyBackend(), recompute_previous_ensemble: bool = False, print_stats: bool = False):
        self.recompute_previous_ensemble = recompute_previous_ensemble
        # Prepare circuits, simulator
        G, gamma, beta = get_qaoa_params(n = self.n, p = self.p, d = self.d)
        self._get_circuits(G, gamma, beta)
        noise_sim = NoiseSimulator(self.noise_model_qtensor, bucket_backend=backend)
        exact_sim = QtreeSimulator(backend=backend)

        # Run simulation
        for num_circs, i in zip(self.num_circs_list, range(len(self.num_circs_list))):
            result = NoiseSimComparisonResult(self.qiskit_circ, self.qtensor_circ, self.noise_model_qiskit, 
                self.noise_model_qtensor, self.n, self.p, self.d, backend)

            if i == 0 or recompute_previous_ensemble == False:
                self.num_circs_simulated.append(num_circs)
                noise_sim.simulate_batch_ensemble(sum(self.qtensor_circ, []), num_circs, self.num_qubits)
                self.qtensor_probs = noise_sim.normalized_ensemble_probs
            else: 
                actual_num_circs = num_circs - self.num_circs_list[i - 1]
                self.num_circs_simulated.append(actual_num_circs)
                noise_sim.simulate_batch_ensemble(sum(self.qtensor_circ, []), actual_num_circs, self.num_qubits)
                prev_qtensor_probs = self.prev_probs
                curr_qtensor_probs = noise_sim.normalized_ensemble_probs
                self.qtensor_probs = (curr_qtensor_probs + prev_qtensor_probs) / 2
            
            if recompute_previous_ensemble == True:
                self.prev_probs = self.qtensor_probs

            qtensor_sim_time = noise_sim.time_taken
            self.simulate_qiskit_density_matrix(self.qiskit_circ, self.noise_model_qiskit)    
            self.exact_qtensor_amps = exact_sim.simulate_batch(sum(self.qtensor_circ, []), batch_vars=self.num_qubits)

            # Save results
            result.save_result(self.qiskit_probs, self.qtensor_probs, self.exact_qtensor_amps, num_circs,
                self.num_circs_simulated[i], G, gamma, beta, qtensor_sim_time, self.qiskit_sim_time)
            self.results.append(result.data)
            if print_stats:
                result.print_result()

    def _mpi_parallel_unit(self, args):
        noise_sim, qtensor_circ, num_circs, num_qubits = args
        noise_sim.simulate_batch_ensemble(sum(qtensor_circ, []), num_circs, num_qubits)
        fraction_of_qtensor_probs = noise_sim.normalized_ensemble_probs
        return fraction_of_qtensor_probs

    def qtensor_qiskit_noisy_qaoa_mpi(self,  num_nodes: int, num_jobs_per_node: int, backend = NumpyBackend(), recompute_previous_ensemble: bool = False, print_stats: bool = True, pbar: bool = True):
        self.num_nodes = num_nodes 
        self.num_jobs_per_node = num_jobs_per_node
        self.recompute_previous_ensemble = recompute_previous_ensemble

        # Prepare circuit, simulator, and area to save results, 
        G, gamma, beta = get_qaoa_params(n = self.n, p = self.p, d = self.d)
        self._get_circuits(G, gamma, beta)
        self.noise_sim = NoiseSimulator(self.noise_model_qtensor, bucket_backend=backend)
        exact_sim = QtreeSimulator(backend=backend)

        for num_circs, i in zip(self.num_circs_list, range(len(self.num_circs_list))):
            result = NoiseSimComparisonResult(self.qiskit_circ, self.qtensor_circ, self.noise_model_qiskit, 
                self.noise_model_qtensor, self.n, self.p, self.d, backend)
            self._get_args(i)
            qtensor_probs_list = tools.mpi.mpi_map(self._mpi_parallel_unit, self._arggen, pbar=pbar, total=num_nodes)
            if qtensor_probs_list:
                if i == 0 or recompute_previous_ensemble == False: 
                    self.qtensor_probs = sum(qtensor_probs_list) / self._total_jobs
                else:
                    prev_qtensor_probs = self.prev_probs
                    curr_qtensor_probs = sum(qtensor_probs_list) / self._total_jobs
                    self.qtensor_probs = (curr_qtensor_probs + prev_qtensor_probs) / 2
                qtensor_sim_time = self.noise_sim.time_taken
                if recompute_previous_ensemble == True:
                    self._check_correct_num_circs_simulated(i)
                    self.prev_probs = self.qtensor_probs

                self.simulate_qiskit_density_matrix(self.qiskit_circ, self.noise_model_qiskit, backend)
                self.exact_qtensor_amps = exact_sim.simulate_batch(sum(self.qtensor_circ, []), batch_vars=self.num_qubits)
                # Save results 
                result.save_result(self.qiskit_probs, self.qtensor_probs, self.exact_qtensor_amps, num_circs,
                    self.num_circs_simulated[i], G, gamma, beta, qtensor_sim_time, self.qiskit_sim_time)
                self.results.append(result.data) 
                if print_stats:
                    tools.mpi.print_stats()
                    result.print_result()

    def qtensor_qiskit_noisy_qaoa_density(self, backend = NumpyBackend(), recompute_previous_ensemble: bool = False):
        self.recompute_previous_ensemble = recompute_previous_ensemble
        # Prepare circuits, simulator, and area to save results
        G, gamma, beta = get_qaoa_params(n = self.n, p = self.p, d = self.d)
        self._get_circuits(G, gamma, beta)
        noise_sim = NoiseSimulator(self.noise_model_qtensor, bucket_backend=backend)
        
        # Simulate
        for num_circs, i in zip(self.num_circs_list, range(len(self.num_circs_list))):
            result = NoiseSimComparisonResult(self.qiskit_circ, self.qtensor_circ, self.noise_model_qiskit, 
                self.noise_model_qtensor, self.n, self.p, self.d)
            if i == 0 or recompute_previous_ensemble == False: 
                self.num_circs_simulated.append(num_circs)
                noise_sim.simulate_batch_ensemble_density(self.qtensor_circ, num_circs, self.n)
                self.qtensor_density_matrix = noise_sim.normalized_ensemble_density_matrix
            else: 
                actual_num_circs = num_circs - self.num_circs_list[i - 1]
                self.num_circs_simulated.append(actual_num_circs)
                noise_sim.simulate_batch_ensemble_density(self.qtensor_circ, actual_num_circs, self.n)
                prev_density_matrix = self.prev_qtensor_density_matrix
                curr_density_matrix = noise_sim.normalized_ensemble_probs
                self.qtensor_density_matrix = (curr_density_matrix + prev_density_matrix) / 2
            qtensor_sim_time = noise_sim.time_taken
            self.simulate_qiskit_density_matrix(self.qiskit_circ, self.noise_model_qiskit, take_trace = False)

            if recompute_previous_ensemble == False: 
                self.prev_qtensor_density_matrix = self.qtensor_density_matrix
            #Save results
            result.save_results_density(self.qiskit_density_matrix, self.qtensor_density_matrix, num_circs, 
                self.num_circs_simulated[i], G, gamma, beta, qtensor_sim_time, self.qiskit_sim_time)
            self.results.append(result.data)


    # Prepare arguments to be sent to each unit of work 
    def _get_args(self, i):
        if i == 0 or self.recompute_previous_ensemble == False:
            num_circs = self.num_circs_list[i]
        else:
            num_circs = self.num_circs_list[i] - self.num_circs_list[i - 1]
        min_circs_per_job = min(10, num_circs)
        if self.num_nodes * self.num_jobs_per_node > num_circs / min_circs_per_job:
            num_circs_per_job = min_circs_per_job
            total_jobs = int(np.ceil(num_circs / num_circs_per_job))
        else: 
            total_jobs = self.num_nodes * self.num_jobs_per_node
            num_circs_per_job = int(np.floor(num_circs / total_jobs))

        ## We make sure that regardless of how many nodes and jobs per node we have, we always 
        ## simulate the exact number of circuits in the ensemble specified. 
        if num_circs_per_job * total_jobs == num_circs:
            self._arggen = list(zip(repeat(self.noise_sim, total_jobs), repeat(self.qtensor_circ, total_jobs), 
                repeat(num_circs_per_job, total_jobs), repeat(self.n, total_jobs)))
            self._total_jobs = total_jobs
            self.num_circs_simulated.append(num_circs)
        else: 
            if num_circs_per_job == min_circs_per_job:
                self._arggen = list(zip(repeat(self.noise_sim, total_jobs - 1), repeat(self.qtensor_circ, total_jobs - 1), 
                    repeat(num_circs_per_job, total_jobs - 1), repeat(self.n, total_jobs - 1)))
                num_circs_in_last_job = num_circs % num_circs_per_job
                actual_num_circs = (total_jobs - 1) * num_circs_per_job + num_circs_in_last_job
                self._arggen.append((self.noise_sim, self.qtensor_circ, num_circs_in_last_job, self.n))
            else: 
                first_set_of_jobs = num_circs - (total_jobs * num_circs_per_job)
                second_set_of_jobs = total_jobs - first_set_of_jobs
                self._arggen = list(zip(repeat(self.noise_sim, first_set_of_jobs), repeat(self.qtensor_circ,  first_set_of_jobs), 
                    repeat(num_circs_per_job + 1,  first_set_of_jobs), repeat(self.n, first_set_of_jobs)))
                self._arggen.extend(list(zip(repeat(self.noise_sim, second_set_of_jobs), repeat(self.qtensor_circ,  second_set_of_jobs), 
                    repeat(num_circs_per_job,  second_set_of_jobs), repeat(self.n, second_set_of_jobs))))  
                actual_num_circs = first_set_of_jobs * (num_circs_per_job + 1) + second_set_of_jobs * num_circs_per_job
            self._total_jobs = total_jobs
            assert num_circs == actual_num_circs
            self.num_circs_simulated.append(actual_num_circs)


    def simulate_qiskit_density_matrix(self, circuit, noise_model_qiskit, backend, take_trace = True):
        start = time.time_ns() / (10 ** 9)
        if isinstance(backend, NumpyBackend):
            qiskit_backend = AerSimulator(method='density_matrix', noise_model=noise_model_qiskit, fusion_enable=False, fusion_verbose=True)
        elif isinstance(backend, CuPyBackend):
            try: 
                qiskit_backend = AerSimulator(method='density_matrix', noise_model=noise_model_qiskit, fusion_enable=False, fusion_verbose=True)
                # qiskit_backend.set_options(device='GPU')
            except AerError as e:
                print(e)
        result = execute(circuit, qiskit_backend, shots=1).result()
        result = qiskit_backend.run(circuit).result()
        if take_trace:
            if isinstance(backend, NumpyBackend):
                self.qiskit_probs = np.diagonal(result.results[0].data.density_matrix.real)
            elif isinstance(backend, CuPyBackend):
                ## TODO: Simulate Qiskit on GPU instead. copying from CPU to GPU will have a serious performance cost
                self.qiskit_probs = cp.diagonal(cp.array(result.results[0].data.density_matrix.real))
        else:
            self.qiskit_density_matrix = result.results[0].data.density_matrix.real
        end = time.time_ns() / (10 ** 9)
        self.qiskit_sim_time = end - start

    def _get_circuits(self, G, gamma, beta):
        # Create Qiskit circuit 
        qiskit_com = QiskitQAOAComposer(graph=G, gamma=gamma, beta=beta)
        qiskit_com.ansatz_state()

        # Convert Qiskit circuit to Qtree circuit
        self.num_qubits, self.qtensor_circ = from_qiskit_circuit(qiskit_com.circuit)
        
        # Finish building remaining portion of Qiskit circuit used only in an Aer simulation 
        qiskit_com.circuit = qiskit_com.circuit.reverse_bits()
        qiskit_com.circuit.save_density_matrix()
        qiskit_com.circuit.measure_all(add_bits = False)
        self.qiskit_circ = qiskit_com.circuit

    def _check_params(self):
        if not isinstance(self.n, int):
            raise Exception("n must an integer.")
        if not isinstance(self.p, int):
            raise Exception("p must an integer.")
        if not isinstance(self.d, int):
            raise Exception("d must an integer.")
        if not isinstance(self.noise_model_qiskit, noise.NoiseModel):
            raise Exception("Qiskit noise model must be of type 'noisel.NoiseModel'")
        if not isinstance(self.noise_model_qtensor, NoiseModel):
            raise Exception("QTensor noise model must be of type NoiseModel.NoiseModel")
        if not isinstance(self.num_circs_list, list):
            raise Exception("The number of circuits must be given as a list. I.e. if num_circs = 10, the argument should be [10].")
        if any(not isinstance(y, int) for y in self.num_circs_list):
            raise Exception("The number of circuits specified must a list of integers.")
        if (self.n * self.d) % 2 != 0:
            raise Exception("n * d must be even. This was not satisfied for the given values d: {}, n: {}".format(self.d, self.n))
        if not 0 <= self.d < self.n:
            raise Exception("The inequality 0 <= d < n was not satisfied for the given values d: {}, n: {}".format(self.d, self.n))

    def _check_correct_num_circs_simulated(self, i):
        if i > 0:
            assert self.num_circs_list[i] == self.num_circs_list[i - 1] + self.num_circs_simulated[i]


