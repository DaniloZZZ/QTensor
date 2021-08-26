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
from cupy import cutensor as cupy_cutensor

class LasyModule:
    def __init__(self, modulename):
        self.modulename = modulename
        self.module = None
    def __getattr__(self, attr):
        if self.module is None:
            self.module = importlib.import_module(self.modulename)
        return self.module.__getattribute__(attr)

np = LasyModule('numpy')
torch = LasyModule('torch')
cupy = LasyModule('cupy')

import sys
import os
sys.path.append(os.environ['HOME']+"/.local")
exatn = LasyModule('exatn')

@dataclass
class BenchResult:
    gen_time: float
    transfer_time: float
    operation_time: float

class Backend():
    @staticmethod
    def prepare(x):
        return x

    @staticmethod
    def get_result(x):
        return x

    timing=pyrofiler.timing

    @classmethod
    def gen_tensors(cls, num_tensors, *size, dtype='float', contraction=''):
        tensors = []
        assert num_tensors == len(size), "number of tensors does not match input size array"
        for i in range(num_tensors):
            tensor = cls.gen_tensor(*size[i], dtype=dtype)
            tensors.append(tensor)
        return tensors

    @classmethod
    def gen_tensor(cls, *size, dtype='float'):
        raise NotImplementedError
    
    @classmethod
    def update_params(self, num_tensors, *size):
        return num_tensors, *size
    
    @classmethod
    def get_operation(cls, task_type):
        if task_type == "matmul":
            return cls.get_matmul()
        elif task_type == "tncontract":
            return cls.get_tncontract()

    @classmethod
    def benchmark(cls, task_type, num_tensors, *size, dtype='float', contraction=''):
        # this line will also trigger lazy import
        num_tensors, size = cls.update_params(num_tensors, *size)
        operation = cls.get_operation(task_type)
        with cls.timing(callback=lambda x: None) as gen:
            tensors = cls.gen_tensors(num_tensors, *size, dtype=dtype, contraction=contraction)
        with cls.timing(callback=lambda x: None) as prep:
            for i in range(len(tensors)):
                tensors[i] = cls.prepare(tensors[i])
        with cls.timing(callback=lambda x: None) as op:
            out_tensor = operation(contraction,*tensors)
        with cls.timing(callback=lambda x: None) as get:
            zr = cls.get_result(out_tensor)
        return zr, BenchResult(gen_time=gen.result, transfer_time=prep.result+get.result, operation_time=op.result)


class Numpy(Backend):
    @staticmethod
    def get_dtype(dtype):
        return {
            'float':np.float32
            ,'double': np.float64
            ,'complex64': np.complex64
            ,'complex128': np.complex128
        }[dtype]

    @classmethod
    def gen_tensor(cls, *sizes, dtype='float'):
        dtype_t = cls.get_dtype(dtype)
        return np.random.rand(*sizes).astype(dtype_t)

    @staticmethod
    def get_matmul():
        return np.matmul
    
    @staticmethod
    def get_tncontract():
        return np.einsum



class Torch(Backend):
    @staticmethod
    def get_dtype(dtype):
        return {
            'float':torch.float32
            ,'double': torch.float64
            ,'complex64': torch.complex64
            ,'complex128': torch.complex128
        }[dtype]

    @classmethod
    def gen_tensor(cls, *sizes, dtype='float'):
        dtype = cls.get_dtype(dtype)
        return torch.rand(*sizes, dtype=dtype)

    @staticmethod
    def get_matmul():
        return torch.matmul
    
    @staticmethod
    def get_tncontract():
        return torch.einsum

class TorchCuda(Torch):
    @classmethod
    @contextmanager
    def timing(cls, **kwargs):
        class Foo:
            pass
        res = Foo()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield res
        end.record()
        torch.cuda.synchronize()
        res.result = start.elapsed_time(end)/1000

    @staticmethod
    def prepare(x):
        return x.to('cuda')



class Cupy(Backend):
    @classmethod
    @contextmanager
    def timing(cls, **kwargs):
        class Foo:
            pass
        res = Foo()
        start = cupy.cuda.Event(disable_timing=False)
        end = cupy.cuda.Event(disable_timing=False)
        start.record()
        yield res
        end.record()

        #I'm not sure about this line, just guessed by analogy from torch
        # Without it raises DeviceNotReady erorr
        end.synchronize()

        res.result = cupy.cuda.get_elapsed_time(start, end)/1000

    @staticmethod
    def get_dtype(dtype):
        return {
            'float':cupy.float32
            ,'double': cupy.float64
            ,'complex64': cupy.complex64
            ,'complex128': cupy.complex128
        }[dtype]

    @classmethod
    def gen_tensor(cls, *sizes, dtype='float'):
        dtype_t = cls.get_dtype(dtype)
        return cupy.random.rand(*sizes).astype(dtype_t)

    @staticmethod
    def get_matmul():
        with pyrofiler.timing('cblas handler'):
            cupy.cuda.device.get_cublas_handle()
        return cupy.matmul
    
    @staticmethod
    def get_tncontract():
        with pyrofiler.timing('cblas handler'): #??
            cupy.cuda.device.get_cublas_handle()
        return cupy.einsum



class CuTensor(Cupy):
    @classmethod
    def gen_tensor(cls, *sizes, dtype='float'):
        dtype_t = cls.get_dtype(dtype)
        return cupy.random.rand(*sizes).astype(dtype_t)
    
    @classmethod
    def update_params(self, num_tensors, *size):
        size = size[0]
        num_tensors += 1
        unit_size = size[0][0]
        size.append([unit_size for i in range(3)])  # Hardcode
        return num_tensors, size

    @classmethod
    def tncontract(cls, contraction, *tensors):
        [x, desc_x] = tensors[0]
        [y, desc_y] = tensors[1]
        [z, desc_z] = tensors[2]
        return cupy_cutensor.contraction(1.0, x, desc_x, ('A', 'B', 'C', 'D'), 
                        y, desc_y, ('B', 'C', 'D', 'F'), 0, 
                        z, desc_z, ('A', 'C', 'F'))
    
    @classmethod
    def get_matmul(cls):
        with pyrofiler.timing('cblas handler'):
            cupy.cuda.device.get_cublas_handle()
        return cls.tncontract
    
    @classmethod
    def get_tncontract(cls):
        with pyrofiler.timing('cblas handler'):
            cupy.cuda.device.get_cublas_handle()
        return cls.tncontract
    
    @classmethod
    def prepare(cls, x):
        desc_x = cupy_cutensor.create_tensor_descriptor(x)
        return [x, desc_x]



# class Operation:
#     def __init__(self, backend:Backend):
#         self.operation = None
#         pass

# class Matmul(Operation):
#     def __init__(self, backend:Backend):
#         self.operation = backend.get_matmul

# class Tncontract(Operation):
#     def __init__(self, backend:Backend):
#         self.operation = backend.get_tncontract



def format_flops(flops):
    ord = 3*int(np.log10(flops)/3)
    suffix = {
        3: 'k'
        ,6: 'M'
        ,9: 'G'
        , 12: 'T'
    }[ord]
    return f'{(flops/10**ord).round(2)}{suffix}'

def obj2dict(obj):
    keys = [x for x in dir(obj) if x[0]!='_']
    return dict((key, obj.__getattribute__(key)) for key in keys)

@lru_cache
def get_gpu_props_json():
    try:
        import torch
        devprops = torch.cuda.get_device_properties(torch.cuda.current_device())
        return obj2dict(devprops)
    except:
        return None

def get_dtype_size(dtype):
    dtype_t = Numpy.get_dtype(dtype)
    x = np.ones(10, dtype=dtype_t)
    return x.itemsize


def mean_mmax(x: list):
    mx, mn = max(x), min(x)
    x.remove(mx)
    x.remove(mn)
    return np.mean(x)

#whether to use the removal of max and min before mean
# does not affect standard deviation or other times, only matmul
use_strip = True

def print_results_json(task_type, backend, *size, dtype, results: List[BenchResult], experiment_group="default group"):
    import json
    GPU_PROPS = get_gpu_props_json()
    tt1 = [r.gen_time for r in results]
    tt2 = [r.operation_time for r in results]
    tt3 = [r.transfer_time for r in results]
    if use_strip:
        m1 = mean_mmax(tt1)
        m2 = mean_mmax(tt2)
        m3 = mean_mmax(tt3)
    else:
        m1, m2, m3 = np.mean(tt1), np.mean(tt2), np.mean(tt3)
    s1, s2, s3 = np.std(tt1), np.std(tt2), np.std(tt3)
    size = size[0]
    if task_type == "matmul":
        pass

    elif task_type == "tncontract":

        # 'abcd,bcdf->acf'
        # size = [[n, n, n, n],
        #          [n, n, n, n]]
        # ops = n**5
        # bytes = dtype_size 2 * n**4

        ops = np.prod(size[0]) * size[1][3]
        size = size[0][0]
        bytes = get_dtype_size(dtype)*2*size**4
        
    ops = ops.item()
    flops = ops/m2

    res = dict(
        task_type=task_type
        , backend=backend
        , size=size
        , itemsize=get_dtype_size(dtype)
        , bytes=bytes 
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
        , flops_str=format_flops(flops)
        , experiment_group=experiment_group
        , fbratio=ops/bytes
    )
    print(json.dumps(res), flush=True)
    return res




def main():

    experiment_group = "Angela_pod_tncontract_restructure_test2"

    task_types = [
        # "matmul",
        "tncontract"
    ]
    contraction = 'abcd,bcdf->acf' # tensor

    # Backend
    backends = {
        'numpy':Numpy
        # , 'exatn': Exatn
    }
    if get_gpu_props_json():
        backends.update({
            'torch':TorchCuda
            ,  'cupy':Cupy
            , 'cutensor': CuTensor
        })
    
    # Tensor properties
    num_tensors = 2
    dim = 4 # tensor
    sizes = [2, 4, 8, 10, 16, 20, 30, 32, 40, 50, 60, 64, 70, 80, 100, 120, 128, 130, 150]  # tensor
    # dim = 2 # matrix
    # sizes = [10, 100, 1000, 1024, 1025, 2000, 4090, 4096] # matrix  
    dtypes = ['float', 'double', 'complex64', 'complex128']

    # Test properties
    repeats = 5
    use_strip = True
    if use_strip:
        repeats += 2
    
    # Bechmark
    import json
    with open('data.json', mode='w') as f:
        for task in task_types:
            for backend in backends:
                for size in sizes:
                    input_sizes = [size for i in range(dim)] # square tensors
                    results = []
                    for dtype in dtypes:
                        for _ in range(repeats):
                            b = backends[backend]
                            _, bench_result = b.benchmark(task, num_tensors, [input_sizes, input_sizes], dtype=dtype, contraction=contraction)
                            results.append(bench_result)
                        json_result = print_results_json(task, backend, [input_sizes, input_sizes], dtype=dtype, results=results, experiment_group=experiment_group)
                          
                        # f.write(json.dumps(json_result))
                        # f.write(",")


if __name__ == "__main__":
    main()

