#from torch._C import device
from .base_class import ContractionBackend
from .numpy import NumpyBackend
from .torch import TorchBackend
from .cupy import CuPyBackend
from .mkl import CMKLExtendedBackend
from .cupy import CuPyBackend
from .transposed import TransposedBackend
from .opt_einsum import OptEinusmBackend
from .transpose_backend import NumpyTranspoedBackend, TorchTransposedBackend, CupyTransposedBackend, CutensorTransposedBackend
from .performance_measurement_decorator import PerfNumpyBackend, PerfBackend, GPUPerfBackend

def get_backend(name):
    backend_dict = {
        'einsum':NumpyBackend,
        'torch_cpu': TorchBackend,
        'torch_gpu': TorchBackend,
        'mkl': CMKLExtendedBackend,
        'tr_einsum': NumpyTranspoedBackend,
        'opt_einsum': OptEinusmBackend,
        'tr_torch': TorchTransposedBackend,
        'cupy': CuPyBackend,
        'tr_cupy': CupyTransposedBackend,
        'tr_cutensor': CutensorTransposedBackend
    }[name]()
    if name in ["torch_gpu", "tr_torch"]:
        return backend_dict['torch'](device = name[-3:])
    else:
        return backend_dict[name]()

def get_cpu_perf_backend(name):
    class MyPerfBackend(PerfBackend):
        Backend = {
            'einsum':NumpyBackend,
            'torch_cpu': TorchBackend,
            'mkl':CMKLExtendedBackend,
            'opt_einsum': OptEinusmBackend,
            'tr_einsum': NumpyTranspoedBackend,
            'opt_einsum': OptEinusmBackend,
        }[name]

    return MyPerfBackend()

def get_gpu_perf_backend(name):
    class MyPerfBackend(GPUPerfBackend):
        Backend = {
            'torch_gpu': TorchBackend,
            'cupy': CuPyBackend,
            'tr_torch': TorchTransposedBackend,
            'tr_cupy': CupyTransposedBackend,
            'tr_cutensor': CutensorTransposedBackend
        }[name]

    if name == "torch_gpu":
        return MyPerfBackend(device="gpu")
    else:
        return MyPerfBackend()
