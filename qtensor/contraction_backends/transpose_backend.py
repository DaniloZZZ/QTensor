from qtensor.contraction_backends import TransposedBackend
from qtree import np_framework
import numpy as np
import torch as torch
import cupy as cp
from cupy import cutensor as cupy_cutensor

class NumpyTranspoedBackend(TransposedBackend):
    def __init__(self):
        super().__init__()
        self.backend = 'numpy'
        self.device = 'cpu'
    
    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        return np_framework.get_sliced_np_buckets(buckets, data_dict, slice_dict)

    @staticmethod
    def get_tncontract():
        return np.einsum
    
    @staticmethod
    def get_sum(data, axis):
        return np.sum(data,axis=axis)
    
    @staticmethod
    def get_transpose(data, *axis):
        return data.transpose(*axis)
    
    @staticmethod
    def get_argsort(*args):
        return np.argsort(*args)


class TorchTransposedBackend(TransposedBackend):
    def __init__(self, device='gpu'):
        super().__init__()
        self.backend_module = torch
        self.backend = 'pytorch'
    
    @staticmethod
    def get_tncontract():
        return torch.einsum
    
    @staticmethod
    def get_sum(data, axis):
        return torch.sum(data,dim=axis)
    
    @staticmethod
    def get_transpose(data, *axis):
        if len(axis) == 0:
            axis = (0,) # Hardcode
        return data.permute(*axis)
    
    @staticmethod
    def get_argsort(*args):
        return torch.argsort(torch.Tensor(*args))

    def prepare(self, data):
        if not isinstance(data, torch.Tensor):
            if self.device == 'gpu' and torch.cuda.is_available():
                cuda = torch.device('cuda')
                data = torch.from_numpy(data).to(cuda)
            else:
                data = torch.from_numpy(data)
        return data



class CupyTransposedBackend(TransposedBackend):
    def __init__(self, device='gpu'):
        super().__init__()
        self.mempool = cp.get_default_memory_pool()
        self.backend_module = cp
        self.backend = 'cupy'
    
    @staticmethod
    def get_tncontract():
        return cp.einsum
    
    @staticmethod
    def get_sum(data, axis):
        return cp.sum(data,axis=axis)
    
    @staticmethod
    def get_transpose(data, *axis):
        data = cp.asarray(data)
        return data.transpose(*axis)
    
    @staticmethod
    def get_argsort(*args):
        return cp.argsort(cp.asarray(*args)).tolist()
    
    def prepare(self, data):
        if self.device == 'gpu':
            data = cp.asarray(data)
        return data


class CutensorTransposedBackend(CupyTransposedBackend):
    def __init__(self, device='gpu'):
        super().__init__()
        self.backend = 'cutensor'
    
    @staticmethod
    def get_tncontract():
        return cupy_cutensor.contraction
    
    @staticmethod
    def get_sum(data, axis):
        return cp.sum(data,axis=axis)
    
    @staticmethod
    def get_transpose(data, *axis):
        return data.transpose(*axis)
    
    @staticmethod
    def get_ready(contraction, a, b):

        # get ready
        inp, out = contraction.split('->')
        size = inp.split(',')
        mode_a = tuple(size[0])
        mode_b = tuple(size[1])
        mode_c = tuple(out)
        
        # generate tensor c
        shape_a, shape_b = a.shape, b.shape
        shape_c = tuple((shape_a[1], shape_a[2], shape_b[2]))
        c = cp.random.rand(*shape_c).astype(a.dtype)

        # manually cast b to a's type
        b = b.astype(a.dtype)

        # generate tensor descriptor
        desc_a = cupy_cutensor.create_tensor_descriptor(a)
        desc_b = cupy_cutensor.create_tensor_descriptor(b)
        desc_c = cupy_cutensor.create_tensor_descriptor(c)

        
        if not a.flags['C_CONTIGUOUS']:
            a = cp.ascontiguousarray(a)
        if not b.flags['C_CONTIGUOUS']:
            b = cp.ascontiguousarray(b)

        return a, b, c, mode_a, mode_b, mode_c, desc_a, desc_b, desc_c
