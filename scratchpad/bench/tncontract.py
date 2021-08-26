from types import DynamicClassAttribute
from numpy.core.fromnumeric import shape, size
import pyrofiler
from typing import List
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import lru_cache
import itertools
import platform
import importlib
import os, psutil
import fire

from base import LasyModule, BenchResult, Backend, get_gpu_props_json, Benchmark, Numpy, TorchCuda, Cupy, CuTensor

np = LasyModule('numpy')

import sys
import os
sys.path.append(os.environ['HOME']+"/.local")
exatn = LasyModule('exatn')


@dataclass
class RandomContract:
    is_random: bool
    contraction: str
    num_indices_result: int = 0
    num_contracted_indices: int = 0
    fill_number: int = 2
    is_transpose: bool = False


class TncontractBench(Benchmark):    
    @staticmethod
    def get_task_type():
        return "tncontract"
    
    @classmethod
    def get_params(cls, *sizes, contraction:RandomContract):
        *in_size, out_size = sizes
        unit_size = in_size[0][0]
        ops = unit_size**(contraction.num_indices_result+contraction.num_contracted_indices)
        param_in = np.prod(in_size[0]) + np.prod(in_size[1])
        param_out = np.prod(out_size)
        return ops, param_in.item(), param_out.item(), unit_size

    @staticmethod
    def benchmark(backend:Backend, num_tensors, contraction:RandomContract, *sizes, dtype='float'):
        num_tensors, *size = backend.get_ready(num_tensors, *sizes, contraction=contraction)
        operation = backend.get_tncontract()
        with backend.timing(callback=lambda x: None) as gen:
            tensors = backend.gen_tensors(num_tensors, *size[0], dtype=dtype)
        with backend.timing(callback=lambda x: None) as prep:
                for i in range(num_tensors):
                    tensors[i] = backend.prepare(tensors[i])
        with backend.timing(callback=lambda x: None) as op:
            if contraction.contraction != '':
                out_tensor = operation(contraction.contraction, *tensors)
            else:
                out_tensor = operation(*tensors)
        with backend.timing(callback=lambda x: None) as get:
            zr = backend.get_result(out_tensor)
        return zr, BenchResult(gen_time=gen.result, transfer_time=prep.result+get.result, operation_time=op.result)

    @classmethod
    def print_results_json(cls, use_strip, backend, *sizes, dtype, results: List[BenchResult], experiment_group="default group", contraction:RandomContract=None):
        prefix = {
            'float': 2
            ,'double': 2
            ,'complex64': 8
            ,'complex128': 8
        }[dtype]
        import json
        GPU_PROPS = get_gpu_props_json()
        tt1 = [r.gen_time for r in results]
        tt2 = [r.operation_time for r in results]
        tt3 = [r.transfer_time for r in results]
        m1, m3 = np.mean(tt1), np.mean(tt3)
        if use_strip:
            m1 = cls.mean_mmax(tt1)
            m2 = cls.mean_mmax(tt2)
            m3 = cls.mean_mmax(tt3)
        else:
            m1, m2, m3 = np.mean(tt1), np.mean(tt2), np.mean(tt3)
        s1, s2, s3 = np.std(tt1), np.std(tt2), np.std(tt3)
        sizes = sizes[0]
        ops, param_in, param_out, unit_size = cls.get_params(*sizes, contraction=contraction)
        flops = prefix*ops/m2
        task_type = cls.get_task_type()
        res = dict(
            task_type=task_type
            , backend=backend
            , size=unit_size
            , sizes=sizes
            , size_idx=[len(x) for x in sizes]
            , contraction=dict(contraction=contraction.contraction
                , num_indices_result=contraction.num_indices_result
                , num_contracted_indices=contraction.num_contracted_indices
                , is_transpose=contraction.is_transpose
                )
            , itemsize=cls.get_dtype_size(dtype)
            , input_bytes=cls.get_dtype_size(dtype)*param_in
            , output_bytes=cls.get_dtype_size(dtype)*param_out
            , dtype=dtype
            , device_props=dict(name=platform.node(), gpu=GPU_PROPS)
            , transfer_time=m3
            , transfer_relstd=s3
            , gen_time=m1
            , gen_relstd=s1
            , operation_time=m2
            , operation_relstd=s2
            , ops=ops
            , flops=flops
            , flops_str=cls.format_flops(flops)
            , fbratio=ops/(cls.get_dtype_size(dtype)*param_in)
            , experiment_group=experiment_group
        )
        print(json.dumps(res), flush=True)
        return res


def gen_sizes(is_random, contraction='', fill_number=2, num_total_indices=0):
    CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

    import random
    from itertools import accumulate

    # seed = 10
    # np.random.seed(seed)
    # random.seed(seed)
    
    if not is_random and contraction != '':
        # square tensors of form [n,n,n,n],[n,n,n,n]->[n,n,n]
        inp, out = contraction.split('->')
        size = inp.split(',')
        sizes = [np.full(len(x), fill_number).tolist() for x in size]
        out_size = np.full(len(out), fill_number).tolist()
        randomcontract = RandomContract(is_random, contraction, 3, 2, fill_number)
    else:
        while True:
            num_indices_result = np.random.randint(num_total_indices)
            num_contracted_indices = num_total_indices - num_indices_result
            if num_indices_result > 1 and num_contracted_indices != 0:
                break

        all_indices = list(CHARS[:num_indices_result+num_contracted_indices])
        contracted_indices = list(np.random.permutation(list(all_indices)[:num_contracted_indices]))
        result_indices = list(set(all_indices) - set(contracted_indices))

        # split the result indices into two array, append the array to contracted indices
        # since each index has to be present in at least one tensor
        while True:
            num_result_in_first_tensor = np.random.randint(num_indices_result)
            if num_result_in_first_tensor != 0 and num_indices_result - num_result_in_first_tensor != 0:
                break
        array = [result_indices[:num_result_in_first_tensor],result_indices[num_result_in_first_tensor:]]

        # choices select the common indices in the results
        choices = []
        for i in range(len(array)):
            choice = np.random.randint(len(array[i])+1)
            choices.append(np.random.permutation(array[(i+1)%2])[:choice].tolist())

        dom_ix = [
            contracted_indices + array[i] + choices[i] for i in range(len(array))
        ]

        # filling the array sizes
        size = [len(x) for x in dom_ix]
        sizes = [np.full(size[0], fill_number).tolist(), np.full(size[1], fill_number).tolist()]
        out_size = np.full(num_indices_result,2).tolist()

        contraction = ','.join(
            ''.join(ix) for ix in dom_ix
        ) + '->' + ''.join(result_indices)

        randomcontract = RandomContract(is_random, contraction, num_indices_result, num_contracted_indices, fill_number)

    sizes.append(out_size)
    return sizes, randomcontract


def permute_sizes(contraction:RandomContract, fill_number=2, num_perm=5):
    num_indices_result = contraction.num_indices_result
    num_contracted_indices = contraction.num_contracted_indices
    contraction = contraction.contraction
    inp, out = contraction.split('->')
    size = inp.split(',')
    sizes = [np.full(len(x), fill_number).tolist() for x in size]
    out_size = np.full(len(out), fill_number).tolist()
    sizes.append(out_size)
    perm_indices = []
    import random
    for i in range(len(size)):
        new_sizes = []
        for j in range(num_perm):
            new_str = ''.join(random.sample(size[i],len(size[i])))
            new_sizes.append(new_str)
        perm_indices.append(new_sizes)
    contractions = []
    for idx_a, idx_b in zip(perm_indices[0], perm_indices[1]):
        contract = idx_a + ',' + idx_b + '->' + out
        randomcontract = RandomContract(is_random=True, contraction=contract, num_indices_result=num_indices_result, num_contracted_indices=num_contracted_indices, fill_number=fill_number, is_transpose=True)
        contractions.append(randomcontract)
    return sizes, contractions


def main():

    experiment_group = "Angela_nslb_tncontract"

    # Backend
    backends = {
        'numpy':Numpy
        # , 'exatn': Exatn
    }
    if get_gpu_props_json():
        backends.update({
            'torch':TorchCuda
            , 'cupy':Cupy
            , 'cutensor': CuTensor
        })
    
    # Tensor properties
    num_tensors = 2
    dtypes = ['float', 'double', 'complex64', 'complex128']

    # Test properties
    repeats = 5
    use_strip = True
    if use_strip:
        repeats += 2
    
    is_random = True
    in_contraction = 'abcd,bcdf->acf' # tensor
    max_num_indices = 26
    num_perm=1
    if is_random:
        in_sizes = np.arange(4,max_num_indices)
    else:
        in_sizes = [10, 16, 20, 30, 32, 40, 50, 60, 64, 70, 80, 100]  # tensor
    
    # Bechmark
    for max_size in in_sizes:
        if not is_random:
            fill_number = max_size
            *sizes, out_contraction = gen_sizes(is_random, contraction=in_contraction, fill_number=fill_number)
            contractions = [out_contraction]
        else:
            fill_number = 2
            *size, out_contraction = gen_sizes(is_random, num_total_indices=max_size)
            *sizes, contractions = permute_sizes(out_contraction, num_perm=num_perm, fill_number=fill_number)
            contractions.append(out_contraction)

        for contraction in contractions:
            for backend in backends:
                try:
                    b = backends[backend]
                    tncontractbench = TncontractBench()
                    results = []
                    for dtype in dtypes:
                        for _ in range(repeats):
                            _, bench_result = tncontractbench.benchmark(b,num_tensors, contraction, *sizes, dtype=dtype)
                            results.append(bench_result)
                        json_result = tncontractbench.print_results_json(use_strip, backend, *sizes, dtype=dtype, results=results, experiment_group=experiment_group, contraction=contraction)      
                except Exception as e:
                    print(e, file=sys.stderr)
                    pass

if __name__ == "__main__":
    fire.Fire(main)

