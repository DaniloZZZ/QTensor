import qtree
import functools, itertools
from qtensor.contraction_backends import NumpyBackend
from qtensor import utils
from loguru import logger as log

class TensorNet:
    @property
    def tensors(self):
        return self._tensors

    def slice(self, slice_dict):
        raise NotImplementedError

    def __len__(self):
        return len(self.tensors)

    def get_line_graph(self):
        raise NotImplementedError


class QtreeTensorNet(TensorNet):
    def __init__(self, buckets, data_dict
                 , bra_vars, ket_vars, free_vars=[]
                 , backend=NumpyBackend()):
        self.buckets = buckets
        self.data_dict = data_dict
        self.bra_vars = bra_vars
        self.ket_vars = ket_vars
        self.free_vars = free_vars
        self.backend = backend

    def set_free_qubits(self, free):
        self.free_vars = [self.bra_vars[i] for i in free]
        self.bra_vars = [var for var in self.bra_vars if var not in self.free_vars]

    def simulation_cost(self, peo):
        ignore_vars = [int(x) for x in self.bra_vars + self.ket_vars]
        peo = [int(x) for x in peo if int(x) not in ignore_vars]
        #peo += [int(x) for x in self.free_vars]
        print("Set graph minus set peo", set(self.get_line_graph().nodes) - set(peo))
        g, _ = utils.reorder_graph(self.get_line_graph(), peo)
        mems, flops = qtree.graph_model.get_contraction_costs(g)
        return mems, flops

    @property
    def _tensors(self):
        return sum(self.buckets, [])

    def slice(self, slice_dict):
        sliced_buckets = self.backend.get_sliced_buckets(
            self.buckets, self.data_dict, slice_dict)
        self.buckets = sliced_buckets
        return self.buckets

    def get_line_graph(self):
        ignored_vars = self.bra_vars + self.ket_vars
        graph =  qtree.graph_model.buckets2graph(self.buckets,
                                               ignore_variables=ignored_vars)
        log.debug('Line graph nodes: {}, edges: {}', graph.number_of_nodes(), graph.number_of_edges())
        return graph

    @classmethod
    def from_qtree_gates(cls, qc, init_state=None, **kwargs):
        all_gates = qc
        n_qubits = len(set(sum([g.qubits for g in all_gates], tuple())))
        qtree_circuit = [[g] for g in qc]
        if init_state is None:
            buckets, data_dict, bra_vars, ket_vars = qtree.optimizer.circ2buckets(
                n_qubits, qtree_circuit)
        else:
            buckets, data_dict, bra_vars, ket_vars = circ2buckets_init(
                n_qubits, qtree_circuit, init_vector=init_state)

        tn = cls(buckets, data_dict, bra_vars, ket_vars, **kwargs)
        return tn

    @classmethod
    def from_qsim_file(cls, file, **kwargs):
        n, qc = qtree.operators.read_circuit_file(file)
        all_gates = sum(qc, [])
        tn = cls.from_qtree_gates(all_gates, **kwargs)
        return tn

    @classmethod
    def from_qiskit_circuit(cls, qiskit_circ, **kwargs):
        n, qc = qtree.operators.from_qiskit_circuit(qiskit_circ)
        all_gates = sum(qc, [])
        tn = cls.from_qtree_gates(all_gates, **kwargs)
        return tn


#-- These functions can be members of tn, but I'm not sure 
# that would be a good place for them, maybe something like 
# TensorContractor is a better place
def reorder_tn(tn, peo):
    perm_buckets, perm_dict = qtree.optimizer.reorder_buckets(tn.buckets, peo)
    tn.ket_vars = sorted([perm_dict[idx] for idx in tn.ket_vars], key=str)
    tn.bra_vars = sorted([perm_dict[idx] for idx in tn.bra_vars], key=str)
    tn.buckets = perm_buckets


def slice_tn(tn, initial_state=0, target_state=0):
    slice_dict = qtree.utils.slice_from_bits(initial_state, tn.ket_vars)
    slice_dict.update(qtree.utils.slice_from_bits(target_state, tn.bra_vars))
    slice_dict.update({var: slice(None) for var in tn.free_vars})
    sliced_buckets = tn.backend.get_sliced_buckets(
        tn.buckets, tn.data_dict, slice_dict)
    return sliced_buckets

def repopulate_data(tn, circ):
    for op in circ:
        data_key = (op.name, hash((op.name, tuple(op.parameters.items()))))
        tn.data_dict[data_key] = op.gen_tensor(**op.parameters)

# --

def circ2buckets_init(qubit_count, circuit, init_vector):
    max_depth = len(circuit)

    data_dict = {}

    # Let's build buckets for bucket elimination algorithm.
    # The circuit is built from left to right, as it operates
    # on the ket ( |0> ) from the left. We thus first place
    # the bra ( <x| ) and then put gates in the reverse order

    # Fill the variable `frame`
    layer_variables = [qtree.optimizer.Var(qubit, name=f'o_{qubit}')
                       for qubit in range(qubit_count)]
    current_var_idx = qubit_count

    # Save variables of the bra
    bra_variables = [var for var in layer_variables]
    ## data_dict[psi.data_key] = init_vector

    # Initialize buckets
    for qubit in range(qubit_count):
        buckets = [[] for qubit in range(qubit_count)]

    # add the initial tensor to first variable

    # Place safeguard measurement circuits before and after
    # the circuit
    measurement_circ = [[qtree.operators.M(qubit) for qubit in range(qubit_count)]]

    combined_circ = functools.reduce(
        lambda x, y: itertools.chain(x, y),
        [measurement_circ, reversed(circuit[:max_depth])])

    # Start building the graph in reverse order
    for layer in combined_circ:
        for op in layer:
            # CUSTOM
            # Swap variables on swap gate 
            if isinstance(op, qtree.operators.SWAP):
                q1, q2 = op.qubits
                _v1 = layer_variables[q1]
                layer_variables[q1] = layer_variables[q2]
                layer_variables[q2] = _v1
                continue

            # build the indices of the gate. If gate
            # changes the basis of a qubit, a new variable
            # has to be introduced and current_var_idx is increased.
            # The order of indices
            # is always (a_new, a, b_new, b, ...), as
            # this is how gate tensors are chosen to be stored
            variables = []
            current_var_idx_copy = current_var_idx
            min_var_idx = current_var_idx
            for qubit in op.qubits:
                if qubit in op.changed_qubits:
                    variables.extend(
                        [layer_variables[qubit],
                         qtree.optimizer.Var(current_var_idx_copy)])
                    current_var_idx_copy += 1
                else:
                    variables.extend([layer_variables[qubit]])
                min_var_idx = min(min_var_idx,
                                  int(layer_variables[qubit]))

            data_key = (op.name,
                        hash((op.name, tuple(op.parameters.items()))))
            # Build a tensor
            t = qtree.optimizer.Tensor(op.name, variables,
                       data_key=data_key)

            # Insert tensor data into data dict
            data_dict[data_key] = op.gen_tensor()

            # Append tensor to buckets
            # first_qubit_var = layer_variables[op.qubits[0]]
            buckets[min_var_idx].append(t)

            # Create new buckets and update current variable frame
            for qubit in op.changed_qubits:
                layer_variables[qubit] = qtree.optimizer.Var(current_var_idx)
                buckets.append(
                    []
                )
                current_var_idx += 1

    # Finally go over the qubits, append measurement gates
    # and collect ket variables
    ket_variables = []

    op = qtree.operators.M(0)  # create a single measurement gate object
    data_key = (op.name, hash((op.name, tuple(op.parameters.items()))))
    data_dict.update(
        {data_key: op.gen_tensor()})

    for qubit in range(qubit_count):
        var = layer_variables[qubit]
        new_var = qtree.optimizer.Var(current_var_idx, name=f'i_{qubit}', size=2)
        ket_variables.append(new_var)
        # update buckets and variable `frame`
        buckets[int(var)].append(
            qtree.optimizer.Tensor(op.name,
                   indices=[var, new_var],
                   data_key=data_key)
        )
        buckets.append([])
        layer_variables[qubit] = new_var
        current_var_idx += 1
    # create initial tensor
    psi = qtree.optimizer.Tensor('psi', ket_variables, data_key=id(init_vector),
                                  data=init_vector.reshape([2]*qubit_count))
    data_dict[id(init_vector)] = init_vector.reshape([2]*qubit_count)
    buckets[int(ket_variables[0])].append(psi)

    return buckets, data_dict, bra_variables, ket_variables


