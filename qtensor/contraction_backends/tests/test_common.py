import qtensor
import pytest
from qtensor.contraction_backends.common import slice_numpy_tensor
from qtensor.contraction_backends import (
    CuPyBackend, NumpyBackend, TorchBackend,
    TorchBackendMatm
)
from qtensor.tests import get_test_qaoa_ansatz_circ
from qtensor import QtreeSimulator
from qtensor.contraction_algos import bucket_elimination
import numpy as np
from qtree.optimizer import Var, Tensor

# -- Contraction

def circ2tn(circ):
    return qtensor.optimisation.TensorNet.QtreeTensorNet.from_qtree_gates(circ)

TEST_BACKENDS = [
    #CuPyBackend,
    NumpyBackend,
    TorchBackend,
    TorchBackendMatm,
]

REFERENCE_BACKEND = NumpyBackend()

# ---- Custom TN

@pytest.mark.parametrize("backend_cls", TEST_BACKENDS)
def test_custom_tn_contract(backend_cls):
    buckets = [
        [Tensor('a', indices=(Var(0, size=2), Var(1, size=3), Var(2, size=4)), data_key='a')],
        [],
        [Tensor('b', indices=(Var(2, size=4), Var(3, size=3)), data_key='b')],
        [Tensor('c', indices=(Var(3, size=3), Var(4, size=2)), data_key='c')],
        [],
    ]
    buckets = [
        [
            Tensor('a', indices=(Var(0, size=2), Var(1, size=3), Var(2, size=4)), data_key='a'),
            Tensor('c', indices=(Var(3, size=3), Var(0, size=2)), data_key='c')
         ],
        [],
        [Tensor('b', indices=(Var(2, size=4), Var(3, size=3)), data_key='b')],
        [],
    ]
    data_dict = {
        'a': np.random.rand(2, 3, 4),
        'b': np.random.rand(4, 3),
        'c': np.random.rand(3, 2),
    }
    slice_dict = {
        Var(1, size=3): 1
    }
    bref = REFERENCE_BACKEND
    def contract_buckets(buckets, slice_dict, data_dict, backend):
        sliced_buckets = backend.get_sliced_buckets(buckets, data_dict, slice_dict)
        print('sliced_buckets', sliced_buckets)
        res = bucket_elimination(sliced_buckets, backend, n_var_nosum=0)
        return backend.get_result_data(res)
    ref = contract_buckets([]+buckets, slice_dict, data_dict, bref)
    b = backend_cls()
    ref = np.einsum('ij,jl,li->',
                    data_dict['a'][:, 1,:],
                    data_dict['b'], data_dict['c'])
    res = contract_buckets([]+buckets, slice_dict, data_dict, b)
    assert np.allclose(ref, res)

# ---- QAOA ansatz

@pytest.mark.parametrize("backend_cls", TEST_BACKENDS)
def test_qaoa_ansatz_contract(backend_cls):
    circ = get_test_qaoa_ansatz_circ(p=3)
    sim_ref = QtreeSimulator(backend=REFERENCE_BACKEND)
    sim = QtreeSimulator(backend=backend_cls())
    ref = sim_ref.simulate(circ)
    res = sim.simulate(circ)
    assert np.allclose(ref, res)

# -- Slicing

def test_slice_numpy_tensor():
    shape = (2, 3, 4, 5)
    indices_in = [Var(i, size=s) for i, s in enumerate(shape)]
    data = np.random.rand(*shape)
    data_ref = data.copy()
    slice_dict = {
        indices_in[0]: slice(None),
        indices_in[1]: slice(1, 3),
        indices_in[2]: 1,
        indices_in[3]: slice(3, 4),
    }
    indices_out = [indices_in[3], indices_in[1], indices_in[0]]
    new_data, new_indices = slice_numpy_tensor(
        data, indices_in, indices_out, slice_dict
    )
    assert new_data.shape == (1, 2, 2)
    assert new_indices == indices_out
    assert np.allclose(data,  data_ref)
    assert not np.allclose(new_data , data_ref[:, 1:3, 1, 3:4])
    assert np.allclose(new_data , data_ref[:, 1:3, 1, 3:4].transpose(2, 1, 0))
    assert np.allclose(new_data , data_ref.transpose()[3:4, 1, 1:3, :])

def test_slice_torch_tensor():
    import torch
    shape = (2, 3, 4, 5)
    indices_in = [Var(i, size=s) for i, s in enumerate(shape)]
    data = torch.randn(*shape)
    data_ref = data.clone()
    slice_dict = {
        indices_in[0]: slice(None),
        indices_in[1]: slice(1, 3),
        indices_in[2]: 1,
        indices_in[3]: slice(3, 4),
    }
    indices_out = [indices_in[3], indices_in[1], indices_in[0]]
    new_data, new_indices = slice_numpy_tensor(
        data, indices_in, indices_out, slice_dict
    )
    assert isinstance(new_data, torch.Tensor)
    assert new_data.shape == (1, 2, 2)
    assert new_indices == indices_out
    assert np.allclose(data,  data_ref)
    assert not np.allclose(new_data , data_ref[:, 1:3, 1, 3:4])
    assert np.allclose(new_data , data_ref[:, 1:3, 1, 3:4].permute(2, 1, 0))
