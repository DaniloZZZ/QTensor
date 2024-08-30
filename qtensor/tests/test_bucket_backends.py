from qtensor import QtreeQAOAComposer
from qtensor.contraction_backends import PerfNumpyBackend
from qtensor.contraction_backends import CuPyBackend, NumpyBackend, CompressionBackend
from qtensor.contraction_backends.torch import TorchBackendMatm
from qtensor.compression import NumpyCompressor, CUSZCompressor
from qtensor.Simulate import CirqSimulator, QtreeSimulator
import qtree

import pytest
import qtensor
import numpy as np
import networkx as nx
from qtensor.tests import get_test_problem, get_test_qaoa_ansatz_circ

from qtensor.contraction_algos import is_reverse_order_backend


def test_profiled(capsys):
    G, gamma, beta = get_test_problem()

    composer = QtreeQAOAComposer(graph=G, gamma=[np.pi / 3], beta=[np.pi / 4])
    composer.ansatz_state()

    print(composer.circuit)
    backend = PerfNumpyBackend()
    sim = QtreeSimulator(backend=backend)

    result = sim.simulate(composer.circuit)
    print("Profile results")
    print(backend.gen_report())

    qtree_amp = result

    assert qtree_amp


def test_reverse_order_switch():
    backend = qtensor.contraction_backends.get_backend("torch")
    reverse = is_reverse_order_backend(backend)
    assert reverse

    backend = qtensor.contraction_backends.get_backend("einsum")
    reverse = is_reverse_order_backend(backend)
    assert not reverse


ref_backend_name = "einsum"


@pytest.mark.parametrize(
    "circ",
    [
        get_test_qaoa_ansatz_circ(n=6, p=3),
        get_test_qaoa_ansatz_circ(n=12, p=4),
    ],
)
@pytest.mark.parametrize(
    ["backend", "atol"],
    [
        # NOTE: 04/02/24 temporary disable cupy backend, it is not working on my machine
        # ('cupy', 1e-10),
        ("torch", 1e-10),
        # ('cupy_compressed', 1e-10),
        (TorchBackendMatm(), 1e-10),
        # (CompressionBackend(
        #    CuPyBackend(),
        #    CUSZCompressor(r2r_error=1e-4, r2r_threshold=1e-5),
        #    11 ),
        #    1e-5)
    ],
)
def test_backends(circ, backend, atol):
    ref_backend = qtensor.contraction_backends.get_backend(ref_backend_name)
    if isinstance(backend, str):
        backend = qtensor.contraction_backends.get_backend(backend)
    sim = QtreeSimulator(backend=backend)
    res = sim.simulate(circ)
    sim_ref = QtreeSimulator(backend=ref_backend)
    res_ref = sim_ref.simulate(circ)
    assert np.allclose(res, res_ref, atol=atol)


ref_backend_name = "einsum"


# -- Bucket contraction tests

def contract_bucket(
    indices_list, backend: qtensor.contraction_backends.ContractionBackend, data_dict,
    slice_dict={}
):
    vars_list = [[qtree.optimizer.Var(i, size=2) for i in ix] for ix in indices_list]
    bucket = [
        qtree.optimizer.Tensor(f"T{i}", indices=ix, data_key=i)
        for i, ix in enumerate(vars_list)
    ]
    print(f"bucket: {bucket}")
    # Empty slice, ensure compatible datatype
    slice_dict = {qtree.optimizer.Var(i, size=2): v for i, v in slice_dict.items()}
    buckets = backend.get_sliced_buckets([bucket], data_dict, slice_dict=slice_dict)
    print(f"sliced bucket: {buckets}")
    result = backend.process_bucket(buckets[0])
    return backend.get_result_data(result)


def contract_bucket_einsum(indices_list, data_dict):
    index_strs = ["".join([chr(97 + i) for i in ix]) for ix in indices_list]
    out_indices = []
    for ix in indices_list:
        for i in ix[1:]:
            if i not in out_indices:
                out_indices.append(i)
    expr = ",".join(index_strs) + "->" + "".join([chr(97 + i) for i in out_indices])
    print(f"expr: {expr}")
    res = np.einsum(expr, *[data_dict[i] for i in range(len(indices_list))])
    return res


@pytest.mark.parametrize(
    ["backend", "atol"],
    [
        # NOTE: 04/02/24 temporary disable cupy backend, it is not working on my machine
        # ('cupy', 1e-10),
        (qtensor.contraction_backends.get_backend("einsum"), 1e-10),
        (qtensor.contraction_backends.get_backend("torch"), 1e-10),
        # ('cupy_compressed', 1e-10),
        (TorchBackendMatm(), 1e-10),
        # (CompressionBackend(
        #    CuPyBackend(),
        #    CUSZCompressor(r2r_error=1e-4, r2r_threshold=1e-5),
        #    11 ),
        #    1e-5)
    ],
)
def test_backend_single_bucket_general(backend, atol):
    """

    Test a single bucket contraction with multiple tensors and a single common
    index
    """

    # -- Generate a simple bucket with decreasing number of indices
    n_tensors = 3
    ix_common = 1
    ix_counter = 2
    indices_list = []
    for nown in range(n_tensors, 0, -1):
        tensor_indices = [ix_common] + list(range(ix_counter, ix_counter + nown))
        ix_counter += nown
        indices_list.append(tensor_indices)
    print(f"indices_list: {indices_list}")
    # -- Generate random data for the tensors
    data_dict = {i: np.random.rand(*[2] * len(ix)) for i, ix in enumerate(indices_list)}

    # Test the slicing correctness as well
    slice_dict = {ix_counter - 1: 1}
    res_ref = contract_bucket_einsum(indices_list, data_dict)
    res_ref = res_ref[..., 1]

    res = contract_bucket(indices_list, backend, data_dict, slice_dict=slice_dict)
    assert np.allclose(res, res_ref, atol=atol)

@pytest.mark.parametrize(
    ["backend", "atol"],
    [
        # NOTE: 04/02/24 temporary disable cupy backend, it is not working on my machine
        # ('cupy', 1e-10),
        (qtensor.contraction_backends.get_backend("einsum"), 1e-10),
        (qtensor.contraction_backends.get_backend("torch"), 1e-10),
        # ('cupy_compressed', 1e-10),
        (TorchBackendMatm(), 1e-10),
        # (CompressionBackend(
        #    CuPyBackend(),
        #    CUSZCompressor(r2r_error=1e-4, r2r_threshold=1e-5),
        #    11 ),
        #    1e-5)
    ],
)
def test_backend_single_bucket_one_index(backend, atol):
    """

    Test a single bucket with several tensors sharing a single index
    """

    # Simple bucket with decreasing number of indices
    n_tensors = 3
    ix = 1
    indices_list = [[ix] for _ in range(n_tensors)]
    print(f"indices_list: {indices_list}")
    data_dict = {i: np.random.rand(*[2] * len(ix)) for i, ix in enumerate(indices_list)}
    res_ref = contract_bucket_einsum(indices_list, data_dict)
    res = contract_bucket(indices_list, backend, data_dict)
    assert np.allclose(res, res_ref, atol=atol)

@pytest.mark.parametrize(
    ["backend", "atol"],
    [
        # NOTE: 04/02/24 temporary disable cupy backend, it is not working on my machine
        # ('cupy', 1e-10),
        (qtensor.contraction_backends.get_backend("einsum"), 1e-10),
        (qtensor.contraction_backends.get_backend("torch"), 1e-10),
        # ('cupy_compressed', 1e-10),
        (TorchBackendMatm(), 1e-10),
        # (CompressionBackend(
        #    CuPyBackend(),
        #    CUSZCompressor(r2r_error=1e-4, r2r_threshold=1e-5),
        #    11 ),
        #    1e-5)
    ],
)
def test_backend_single_bucket_trick(backend, atol):
    """

    Test a single bucket with different common indices
    """

    # Simple bucket with decreasing number of indices
    n_tensors = 3
    ix = 1
    indices_list = [[1, 2], [2, 3], [3]]
    print(f"indices_list: {indices_list}")
    data_dict = {i: np.random.rand(*[2] * len(ix)) for i, ix in enumerate(indices_list)}
    res_ref = contract_bucket_einsum(indices_list, data_dict)
    res = contract_bucket(indices_list, backend, data_dict)
    assert np.allclose(res, res_ref, atol=atol)
