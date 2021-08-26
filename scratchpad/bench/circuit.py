import numpy as np
import random
from itertools import accumulate
import pyrofiler
import opt_einsum
import sys
import os, psutil
import fire

import qtree
import qtensor
# try:
#     import ctf
# except ImportError:
#     print('Can\'t import ctf')

ELEMENT_SIZE = np.zeros((1,), dtype=np.complex128).itemsize


def get_backend(backend):
    return {
        'mkl': qtensor.contraction_backends.CMKLExtendedBackend
        , 'einsum': qtensor.contraction_backends.NumpyBackend
        , 'tr_einsum': qtensor.contraction_backends.NumpyTranspoedBackend
        , 'torch': qtensor.contraction_backends.TorchBackend
        , 'opt_einsum': qtensor.contraction_backends.OptEinusmBackend
        , 'tr_torch': qtensor.contraction_backends.TorchTransposedBackend
        , 'cupy': qtensor.contraction_backends.CuPyBackend
        , 'tr_cupy': qtensor.contraction_backends.CupyTransposedBackend
        , 'tr_cutensor': qtensor.contraction_backends.CutensorTransposedBackend

    }[backend]

def callback(x, desc):
    print(desc, f'{x:,}')

@pyrofiler.mem_util('Memory to contract', callback=callback)
def test_mem(N, p=4, backend='einsum', seed=10, ordering='greedy', merged=False):
    """
    Performs a contraction that
    does not include hard optimization task.
    Majority of indices are in ``d`` tensors, and other ``s`` tensors are
    small

    Args:
        K: total number of indices in result
        C: contracted indices
        d: number of dominating tensors
        s: number of small tensors
    """
    np.random.seed(seed)
    random.seed(seed)
    opt = qtensor.toolbox.get_ordering_algo(ordering)
    G = qtensor.toolbox.random_graph(N, type='random', degree=3, seed=seed)
    comp = qtensor.DefaultQAOAComposer(G, gamma=[.1]*p, beta=[.3]*p)
    comp.ansatz_state()
    tn = qtensor.optimisation.TensorNet.QtreeTensorNet.from_qtree_gates(comp.circuit)
    peo, _ = opt.optimize(tn)
    K = opt.treewidth

    #-- Memory estimations
    process = psutil.Process(os.getpid())
    overhead = process.memory_info().rss
    print('Width', K)
    print(f'Overhead: {overhead:,} Bytes')
    print(f'Should take: {ELEMENT_SIZE*2**K:,} Bytes')
    print(f'Total expected Memory: {overhead+ELEMENT_SIZE*2**K:,} Bytes')
    #--
    mems, flops = tn.simulation_cost(peo)
    print(f'Max mem estimation {max(mems)*ELEMENT_SIZE:,} bytes')
    print('Flops estimation', sum(mems))

    #merged_ix, width = qtensor.utils.find_mergeable_indices(perm_peo, bucket_ix)

    with pyrofiler.timing('Time to simulate'):
        backend = get_backend(backend)()
        shapes = []
        mem = []
        def subst_dec(func):
            def subst(*args, **kwargs):
                ret = func(*args, **kwargs)
                shapes.append(ret.data.shape)
                mem.append(ret.data.size)
                return ret
            return subst
        def subst_dec2(func):
            def subst(*args, **kwargs):
                ret = func(*args, **kwargs)
                shapes.append(ret.shape)
                mem.append(ret.size)
                return ret
            return subst
        if not merged:
            backend.process_bucket = subst_dec(backend.process_bucket)
            sim = qtensor.QtreeSimulator(backend=backend)
            sim.simulate_batch(comp.circuit, peo=peo)
        else:
            backend.process_bucket_merged = subst_dec(backend.process_bucket_merged)
            try:
                backend.pairwise_sum_contract = subst_dec2(backend.pairwise_sum_contract)
            except Exception:
                pass
            sim = qtensor.MergedSimulator.MergedSimulator(backend=backend)
            sim.simulate_batch(comp.circuit, peo=peo)

        print('max shape', max(map(len, shapes)))
        # print(f'max size {max(mem)*ELEMENT_SIZE:,} ')
        try:
            print('sum times', sum(backend.times))
            print('all times', sum(backend.times_all))
            print('pre times', sum(backend.times_pre))
            print('post times', sum(backend.times_post))
            print('sort times', sum(backend.times_sre))
        except Exception as e:
            pass

if __name__=="__main__":
    # fire.Fire(test_mem)
    test_mem(N=10, backend='tr_cutensor')
