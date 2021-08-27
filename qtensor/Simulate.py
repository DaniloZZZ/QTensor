import qtree
import cirq
from qtensor.contraction_backends import NumpyBackend, ContractionBackend

from qtensor.optimisation.TensorNet import QtreeTensorNet
from qtensor.optimisation.Optimizer import DefaultOptimizer, Optimizer
from tqdm.auto import tqdm

from loguru import logger as log

from qtensor import utils

class Simulator:
    def __init__(self):
        pass

    def simulate(self, qc):
        """ Factory method """
        raise NotImplementedError()


class QtreeSimulator(Simulator):
    FallbackOptimizer = DefaultOptimizer
    optimizer: Optimizer
    backend: ContractionBackend

    def __init__(self, backend=NumpyBackend(), optimizer=None, max_tw=None):
        self.backend = backend
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = self.FallbackOptimizer()
        self.max_tw = max_tw

    #-- Internal helpers
    def _new_circuit(self, qc):
        self.all_gates = qc

    def _create_buckets(self):
        self.tn = QtreeTensorNet.from_qtree_gates(self.all_gates,
                                                 backend=self.backend)
        self.tn.backend = self.backend

    def _set_free_qubits(self, free_final_qubits):
        self.tn.free_vars = [self.tn.bra_vars[i] for i in free_final_qubits]
        self.tn.bra_vars = [var for var in self.tn.bra_vars if var not in self.tn.free_vars]

    def _optimize_buckets(self):
        self.peo = self.optimize_buckets()

    def _reorder_buckets(self):
        """
        Permutes indices in the tensor network and peo

        Modifies:
            self.tn.ket_vars
            self.tn.bra_vars
            self.peo
            self.tn.buckets

        Returns:
            perm dict {from:to}
        """
        perm_buckets, perm_dict = qtree.optimizer.reorder_buckets(self.tn.buckets, self.peo)
        self.tn.ket_vars = sorted([perm_dict[idx] for idx in self.tn.ket_vars], key=str)
        self.tn.bra_vars = sorted([perm_dict[idx] for idx in self.tn.bra_vars], key=str)
        if self.peo:
            self.peo = [perm_dict[idx] for idx in self.peo]
        self.tn.buckets = perm_buckets
        return perm_dict

    def set_init_state(self, state):
        """
        Set initial state of system.
        Args:
            state (int): index of state in computation basis in big-endian numeration

        Example:
            sets state ...010 as initial state

            >>> simulator.set_init_state(2)
        """

        self._initial_state = state

    def _get_slice_dict(self, initial_state=0, target_state=0):
        if hasattr(self, 'target_state'):
            target_state = self.target_state
        if hasattr(self, '_initial_state'):
            initial_state = self._initial_state
        slice_dict = qtree.utils.slice_from_bits(initial_state, self.tn.ket_vars)
        slice_dict.update(qtree.utils.slice_from_bits(target_state, self.tn.bra_vars))
        #print('bra', self.tn.bra_vars)
        slice_dict.update({var: slice(None) for var in self.tn.free_vars})
        return slice_dict
    #-- 

    def optimize_buckets(self):
        ## Fro Kahypar
        if hasattr(self, 'hypar_tn'):
            peo, self.tn = self.optimizer.optimize(self.tn, self.hypar_tn)
        else:
            peo, self.tn = self.optimizer.optimize(self.tn)
        
        # print('Treewidth', self.optimizer.treewidth)
        # print(peo)
        return peo

    def prepare_buckets(self, qc, batch_vars=0, peo=None):
        self._new_circuit(qc)
        self._create_buckets()
        # Collect free qubit variables
        if isinstance(batch_vars, int):
            free_final_qubits = list(range(batch_vars))
        else:
            free_final_qubits = batch_vars

        self._set_free_qubits(free_final_qubits)
        if peo is None:
            self._optimize_buckets()
            if self.max_tw:
                if self.optimizer.treewidth > self.max_tw:
                    raise ValueError(f'Treewidth {self.optimizer.treewidth} is larger than max_tw={self.max_tw}.')
        else:
            self.peo = peo

        all_indices = sum([list(t.indices) for bucket in self.tn.buckets for t in bucket], [])
        identity_map = {v.name: v for v in all_indices}
        self.peo = [identity_map[i.name] for i in self.peo]

        self._reorder_buckets()
        slice_dict = self._get_slice_dict()
        #log.info('batch slice {}', slice_dict)

        sliced_buckets = self.tn.slice(slice_dict)
        #self.backend.pbar.set_total ( len(sliced_buckets))
        self.buckets = sliced_buckets
        # print("Buckets:")
        # print(sliced_buckets)

    def simulate_batch(self, qc, batch_vars=0, peo=None):
        self.prepare_buckets(qc, batch_vars, peo)

        result = qtree.optimizer.bucket_elimination(
            self.buckets, self.backend.process_bucket,
            n_var_nosum=len(self.tn.free_vars)
        )
        return self.backend.get_result_data(result).flatten()

    def simulate(self, qc):
        return self.simulate_state(qc)

    def simulate_state(self, qc, peo=None):
        return self.simulate_batch(qc, peo=peo, batch_vars=0)

class CirqSimulator(Simulator):

    def simulate(self, qc, **params):
        sim = cirq.Simulator(**params)
        return sim.simulate(qc)

