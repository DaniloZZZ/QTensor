from numpy.core.fromnumeric import size
import pyrofiler
from typing import List
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import itertools
import platform
import importlib

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
cupy_cutensor = LasyModule('cutensor')
import sys
import os
sys.path.append(os.environ['HOME']+"/.local")
exatn = LasyModule('exatn')

@dataclass
class BenchResult:
    gen_time: float
    transfer_time: float
    mult_time: float


class Backend:
    @staticmethod
    def prepare(x):
        return x


    @staticmethod
    def get_result(x):
        return x

    timing=pyrofiler.timing

    @classmethod
    def gen_tensor(cls, *sizes, dtype='float'):
        raise NotImplementedError

    @staticmethod
    def get_matmul():
        raise NotImplementedError

    @classmethod
    def benchmark_matmul(cls, size_m, size_n, size_l, dtype):
        # this line will also trigger lazy import
        matmul = cls.get_matmul()
        with cls.timing(callback=lambda x: None) as gen:
            x = cls.gen_tensor(size_m, size_n, dtype=dtype)
            y = cls.gen_tensor(size_n, size_l, dtype=dtype)
            if cls == CuTensor:
                z = cls.gen_tensor(size_m, size_l, dtype=dtype)
        with cls.timing(callback=lambda x: None) as prep:
            x = cls.prepare(x)
            y = cls.prepare(y)
            if cls == CuTensor:
                z = cls.prepare(z)
        with cls.timing(callback=lambda x: None) as mm:
            if cls == CuTensor:
                z = matmul(x,y,z)
            else:
                z = matmul(x,y)
        with cls.timing(callback=lambda x: None) as get:
            zr = cls.get_result(z)
        return zr, BenchResult(gen_time=gen.result, transfer_time=prep.result+get.result, mult_time=mm.result)

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


class CuTensor(Cupy):

    @classmethod
    def gen_tensor(cls, *sizes, dtype='float'):
        dtype_t = cls.get_dtype(dtype)
        return cupy.random.rand(*sizes).astype(dtype_t)

    @classmethod
    def cutensor_matmul(cls, x, y, z):
        [x, desc_x] = x
        [y, desc_y] = y
        [z, desc_z] = z
        from cupy import cutensor
        return cutensor.contraction(1, x, desc_x, cls.mode_x, 
                                y, desc_y, cls.mode_y, 0, 
                                z, desc_z, cls.mode_z)
    
    @classmethod
    def get_matmul(cls):
        with pyrofiler.timing('cblas handler'):
            cupy.cuda.device.get_cublas_handle()
        return cls.cutensor_matmul
    
    @classmethod
    def prepare(cls, x):
        from cupy import cutensor
        if not hasattr(cls, 'extent'):
            cls.mode_x = ('m', 'k')
            cls.mode_y = ('k', 'n')
            cls.mode_z = ('m', 'n')
            cls.mode_x = cutensor.create_mode(*cls.mode_x)
            cls.mode_y = cutensor.create_mode(*cls.mode_y)
            cls.mode_z = cutensor.create_mode(*cls.mode_z)
        desc_x = cutensor.create_tensor_descriptor(x)
        return [x, desc_x]
    

@dataclass
class ExatnTensor:
    name: str
    shape: tuple
    dtype: str

class Exatn(Backend):
    # infinite name generator
    name_generator = (hex(x)[1:] for x, _
                      in enumerate(itertools.repeat(0)))
    allocated_tensor_names = []
    @classmethod
    def cleanup_tensors(cls):
        for name in cls.allocated_tensor_names:
            # will produce a warning on non-existent tensor
            exatn.destroyTensor(name)
        cls.allocated_tensor_names = []

    @staticmethod
    def get_dtype(dtype):
        return Numpy.get_dtype(dtype)

    @classmethod
    def gen_tensor(cls, *sizes, dtype='float'):
        tname = next(cls.name_generator)
        rand = Numpy.gen_tensor(*sizes, dtype=dtype)
        #print('create ten', tname)
        success = exatn.createTensor(tname, rand.copy(order='F'))
        #print('done', tname)
        if success:
            cls.allocated_tensor_names.append(tname)
            return ExatnTensor(name=tname, shape=sizes, dtype=dtype)

    @classmethod
    def exatn_matmul(cls, x, y):
        """
        Takes two names of tensors, should be already allocated,
        returns name of resulting tensor

        Args:
            x: ExatnTensor
            y: ExatnTensor
        """
        #exatn.evaluateTensorNetwork('sum', 'SR1() = R1(a)*R2(a)')
        res = next(cls.name_generator)
        res_shape = x.shape[0], y.shape[1]
        #print('create res', res, res_shape)
        dtype = Numpy.get_dtype(x.dtype)
        res_body = np.zeros(res_shape, dtype=dtype)
        exatn.createTensor(res, res_body)
        cls.allocated_tensor_names.append(res)
        st = f'{res}(a,c)={x.name}(a,b)*{y.name}(b,c)'
        #print('st', st)
        _ = exatn.contractTensors(st, 1.0)
        #print('contr')
        return res

    @classmethod
    def get_matmul(cls):
        return cls.exatn_matmul

    @classmethod
    def get_result(cls, x):
        t = exatn.getLocalTensor(x)
        cls.cleanup_tensors()
        return t


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


def print_results_csv(backend, size, dtype, results: List[BenchResult]):
    tt1 = [r.gen_time for r in results]
    tt2 = [r.mult_time for r in results]
    m1, m2 = np.mean(tt1), np.mean(tt2)
    s1, s2 = np.std(tt1), np.std(tt2)
    flops = size**3/m2
    print(f'{backend}, {size}, {dtype}, {m1}, {(s1/m1).round(4)}, {m2}, {(s2/m2).round(4)}, {format_flops(flops)}')


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

def print_results_json(task_type, backend, size_m, size_n, size_l, dtype, results: List[BenchResult], experiment_group="default group"):
    import json
    GPU_PROPS = get_gpu_props_json()
    tt1 = [r.gen_time for r in results]
    tt2 = [r.mult_time for r in results]
    tt3 = [r.transfer_time for r in results]
    m1, m3 = np.mean(tt1), np.mean(tt3)
    if use_strip:
        m2 = mean_mmax(tt2)
    else:
        m2 = np.mean(tt2)
    s1, s2, s3 = np.std(tt1), np.std(tt2), np.std(tt3)
    flops = size_m*size_n*size_l/m2
    res = dict(
        task_type=task_type
        , backend=backend
        , size_m=size_m
        , size_n=size_n
        , size_l=size_l
        , itemsize=get_dtype_size(dtype)
        , bytes=get_dtype_size(dtype)*(size_m*size_n+size_n*size_l)
        , dtype=dtype
        , device_props=dict(name=platform.node(), gpu=GPU_PROPS)
        , transfer_time=m3
        , transfer_relstd=s3
        , gen_time=m1
        , gen_relstd=s1
        , mult_time=m2
        , mult_relstd=s2
        , flops=flops
        , flops_str=format_flops(flops)
        , experiment_group=experiment_group
    )
    print(json.dumps(res), flush=True)
    return res


def main():

    sizes_m = [10, 100, 1000, 1024, 1025, 2000, 4090, 4096]
    sizes_n = [10, 100, 1000, 1024, 1025, 2000, 4090, 4096]
    sizes_l = [10, 100, 1000, 1024, 1025, 2000, 4090, 4096]
    # sizes_l = [2, 2, 2, 2, 2, 2, 2, 2]
    experiment_group = "default_pod_matmul"
    #sizes = [2000, 3000]
    backends = {
        'numpy':Numpy
        # ,'exatn': Exatn
    }
    if get_gpu_props_json():
        backends.update({
            'torch':TorchCuda,
            'cupy':Cupy,
            'cutensor': CuTensor
        })

    repeats = 5
    if use_strip:
        repeats += 2
    task_type = 'matmul'
    dtypes = ['float', 'double', 'complex64', 'complex128']

    # print(f'backend, size, dtype, Time1 mean, Time1 relstd, Time2 mean, Time2 relstd, FLOPs')
    
    import json

    with open('data.json', mode='w') as f:

        for backend in backends:
            for size_m, size_n, size_l in zip(sizes_m, sizes_n, sizes_l):
                results = []
                for dtype in dtypes:
                    for _ in range(repeats):
                        b = backends[backend]
                        _, bench_result = b.benchmark_matmul(size_m, size_n, size_l, dtype)
                        results.append(bench_result)

                    json_result = print_results_json(task_type, backend, size_m, size_n, size_l, dtype, results, experiment_group)
                    # f.write(json.dumps(json_result))
                    # f.write(",")

if __name__ == "__main__":
    main()

