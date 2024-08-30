import qtensor
import numpy as np

from qtensor.contraction_backends import CuPyBackend, NumpyBackend

import pytest
cp = pytest.importorskip('cupy')


def get_test_qaoa_circ(n=10, p=2, d=3, type='random'):
    G = qtensor.toolbox.random_graph(seed=10, degree=d, nodes=n, type=type)
    print('Test problem: n, p, d', n, p, d)
    gamma, beta = [np.pi/5]*p, [np.pi/2]*p

    composer = qtensor.DefaultQAOAComposer(
        graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()
    return composer.circuit


def get_test_qaoa_tn(n=10, p=2, d=3, type='random'):
    G = qtensor.toolbox.random_graph(seed=10, degree=d, nodes=n, type=type)
    print('Test problem: n, p, d', n, p, d)
    gamma, beta = [np.pi/5]*p, [np.pi/2]*p

    composer = qtensor.DefaultQAOAComposer(
        graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()
    tn = qtensor.optimisation.TensorNet.QtreeTensorNet.from_qtree_gates(composer.circuit)
    return tn




def test_cupy_process_bucket():
    btr = CuPyBackend()
    bnp = NumpyBackend()
    def contract_tn(backend, search_len=1, test_problem_kwargs={}):
        tn = get_test_qaoa_tn(**test_problem_kwargs)
        sliced_buckets = backend.get_sliced_buckets(tn.buckets, tn.data_dict, {})
        good_buckets = [x for x in sliced_buckets if len(x) >= search_len]
        if not good_buckets:
            raise ValueError('Could not find large enough buckets')
        selected_bucket = good_buckets[0]
        print('selected_bucket', selected_bucket)

        result = backend.process_bucket(selected_bucket)
        return backend.get_result_data(result)

    restr = contract_tn(btr, 1)
    resnp = contract_tn(bnp, 1)
    assert type(restr) is cp.ndarray
    assert np.allclose(restr, resnp)
    restr = contract_tn(btr, 2, dict(n=10, p=4, d=3))
    resnp = contract_tn(bnp, 2, dict(n=10, p=4, d=3))

    assert restr.shape == resnp.shape
    assert np.allclose(restr, resnp)
    
def test_cupy_get_sliced__slice():
    backend = CuPyBackend()
    tn = get_test_qaoa_tn()
    tensor = tn.buckets[0][0]
    tensor_data = tn.data_dict[tensor.data_key]
    slice_dict = {
        tensor.indices[0]: slice(0, 1)
    }
    buckets = backend.get_sliced_buckets(
        tn.buckets, tn.data_dict, slice_dict
    )
    assert tensor_data.shape != buckets[0][0].data.shape
    
    
def test_cupy_get_sliced__noslice():
    backend = CuPyBackend()
    tn = get_test_qaoa_tn()

    tensor = tn.buckets[0][0]
    tensor_data = tn.data_dict[tensor.data_key]
    buckets = backend.get_sliced_buckets(
        tn.buckets, tn.data_dict, {}
    )
    assert len(buckets) == len(tn.buckets)
    assert tensor_data.shape == buckets[0][0].data.shape

def test_cupy_get_sliced_smoke():
    backend = CuPyBackend()
    test_buckets = []
    buckets = backend.get_sliced_buckets(test_buckets, {}, {})
    assert buckets == []
