import numpy as np
import ctypes
from ctypes import *
import random
from qtensor.tools.lazy_import import cupy as cp
import time
import torch

from pathlib import Path
#LIB_PATH = str(Path(__file__).parent/'libcuszx_wrapper.so')
LIB_PATH='/home/mkshah5/QTensor/qtensor/compression/szx/src/libcuszx_wrapper.so'
# unsigned char* cuSZx_integrated_compress(float *data, float r2r_threshold, float r2r_err, size_t nbEle, int blockSize, size_t *outSize)

def get_host_compress():
    dll = ctypes.CDLL(LIB_PATH, mode=ctypes.RTLD_GLOBAL)
    func = dll.cuSZx_integrated_compress
    # Returns: unsigned char *bytes
    # Needs: float *data, float r2r_threshold, float r2r_err, size_t nbEle, int blockSize, size_t *outSize
    func.argtypes = [POINTER(c_float), c_float, c_float, c_size_t, c_int, POINTER(c_size_t)]
    func.restype = POINTER(c_ubyte)
    return func

# float* cuSZx_integrated_decompress(unsigned char *bytes, size_t nbEle)

def get_host_decompress():
    dll = ctypes.CDLL(LIB_PATH, mode=ctypes.RTLD_GLOBAL)
    func = dll.cuSZx_integrated_decompress
    # Returns: float *newData
    # Needs: size_t nbEle, unsigned char *cmpBytes
    func.argtypes = [POINTER(c_ubyte), c_size_t]
    func.restype = POINTER(c_float)
    return func

def get_device_compress():
    dll = ctypes.CDLL(LIB_PATH, mode=ctypes.RTLD_GLOBAL)
    func = dll.cuSZx_device_compress
    # Returns: unsigned char *bytes
    # Needs: float *oriData, size_t *outSize, float absErrBound, size_t nbEle, int blockSize, float threshold
    func.argtypes = [POINTER(c_float), POINTER(c_size_t), c_float, c_size_t, c_int, c_float]
    func.restype = POINTER(c_ubyte)
    return func

def get_device_decompress():
    dll = ctypes.CDLL(LIB_PATH, mode=ctypes.RTLD_GLOBAL)
    func = dll.cuSZx_device_decompress
    # Returns: float *newData
    # Needs: size_t nbEle, unsigned char *cmpBytes
    func.argtypes = [c_size_t, POINTER(c_ubyte)]
    func.restype = POINTER(c_float)
    return func


def cuszx_host_compress(oriData, absErrBound, nbEle, blockSize,threshold):
    __cuszx_host_compress = get_host_compress()

    variable = ctypes.c_size_t(0)
    outSize = ctypes.pointer(variable)

    oriData_p = ctypes.cast(oriD.data.ptr, ctypes.POINTER(c_float))

    o_bytes = __cuszx_host_compress(oriData_p, outSize,np.float32(absErrBound), np.ulonglong(nbEle), np.int32(blockSize),np.float32(threshold))

    return o_bytes, outSize

def cuszx_host_decompress(nbEle, cmpBytes):
    __cuszx_host_decompress=get_host_decompress()

    nbEle_p = ctypes.c_size_t(nbEle)
    newData = __cuszx_host_decompress(nbEle_p,cmpBytes)
    return newData


def cuszx_device_compress(oriData, absErrBound, nbEle, blockSize,threshold):
    __cuszx_device_compress = get_device_compress()
    
    ori_nbEle = nbEle
    variable = ctypes.c_size_t(0)
    outSize = ctypes.pointer(variable)
    #absErrBound = absErrBound*(cp.amax(oriData.get())-cp.amin(oriData.get()))
    #threshold = threshold*(cp.amax(oriData.get())-cp.amin(oriData.get()))
    oriData = oriData.flatten()
    ori_real = oriData.real
    ori_imag = oriData.imag
    oriData = cp.concatenate((ori_real, ori_imag))
   # print(oriData.dtype)
    sample = oriData[::2]
    
    #torch_tensor = torch.as_tensor(sample, device='cuda')
    #d = torch.max(torch_tensor).item() - torch.min(torch_tensor).item()
    #s_sample = cp.sort(sample)
    #d = s_sample[-1] - s_sample[0]
    #v_time = time.time()
    #print(type(oriData))
    d = cp.amax(oriData) - cp.amin(oriData)
    #print("max min time (s): " +str(time.time()-v_time))
    d = d.get()
    if d.dtype == np.complex64:
        #d = min(d.real, d.imag)
        d = d.real
    absErrBound = absErrBound*(d)
    threshold = threshold*(d)
    s_1 = time.time() 
    #print(cp.get_array_module(oriData))    
    truth_values = abs(oriData)<=threshold
    oriData[truth_values] = 0.0
    truth_values = cp.invert(truth_values)
    oriData = oriData[truth_values]
    bitmap = truth_values
    nbEle = oriData.shape[0]
    

    oriData_p = ctypes.cast(oriData.data.ptr, ctypes.POINTER(c_float))
    #print("starting") 
    o_bytes = __cuszx_device_compress(oriData_p, outSize,np.float32(absErrBound), np.ulonglong(nbEle), np.int32(blockSize),np.float32(threshold))
  
    #print("tg and max time (s): "+str(time.time()-s_1))
    #print("bitmap shape: "+str(bitmap.shape[0]))
    #print("percent nonzero bytes: "+str(bitmap[cp.nonzero(bitmap)].shape[0]/bitmap.shape[0]))
    #print("CR")
    print((ori_nbEle*4)/(outSize[0] + bitmap.shape[0]/8))
    return (o_bytes,bitmap), outSize


def cuszx_device_decompress(nbEle, cmpBytes, owner, dtype):
    __cuszx_device_decompress=get_device_decompress()
    (cmpBytes, bitmap) = cmpBytes
    #print("bitmap len:" +str(len(bitmap)))
    #print(nbEle)
    tmp_nbEle = cp.count_nonzero(bitmap).item()
    #print(tmp_nbEle)
    nbEle_p = ctypes.c_size_t(tmp_nbEle)
    newData = __cuszx_device_decompress(nbEle_p,cmpBytes)

    # decompressed_ptr = self.cuszx_decompress(isCuPy, cmp_bytes, num_elements_eff)
    # -- Workaround to convert GPU pointer to int
    p_decompressed_ptr = ctypes.addressof(newData)
    # cast to int64 pointer
    # (effectively converting pointer to pointer to addr to pointer to int64)
    p_decompressed_int= ctypes.cast(p_decompressed_ptr, ctypes.POINTER(ctypes.c_uint64))
    decompressed_int = p_decompressed_int.contents
    # --
    pointer_for_free = decompressed_int.value
    # self.decompressed_own.append(decompressed_int.value)
    mem = cp.cuda.UnownedMemory(decompressed_int.value, tmp_nbEle, owner, device_id=0)
    mem_ptr = cp.cuda.memory.MemoryPointer(mem, 0)
    #print("mem ptr")
    #print(mem_ptr)
    arr = cp.ndarray(shape=(tmp_nbEle,), dtype=np.float32, memptr=mem_ptr)

    res = cp.zeros((nbEle,))
    ## need to convert newData to cupy
    cp.place(res,bitmap,arr)

    c_res = cp.zeros(int(nbEle/2), np.complex64)
    c_res.real = res[0:int(nbEle/2)]
    c_res.imag = res[int(nbEle/2):]
    return (c_res, pointer_for_free)

### Example of device compress/decompress wrapper usage
class Comp():
    def __init__(self):
        self.name = "dummy"

def free_compressed(ptr):
    p_ptr = ctypes.addressof(ptr)
    p_int = ctypes.cast(p_ptr, ctypes.POINTER(ctypes.c_uint64))
    decomp_int = p_int.contents
    cp.cuda.runtime.free(decomp_int.value)


if __name__ == "__main__":
    
    DATA_SIZE = int(1024)
    MAX_D = 10.0
    MIN_D = -10.0
    RANGE = MAX_D - MIN_D
    r2r_threshold = 0.002
    r2r_error = 0.0001

    in_vector = np.fromfile("all_sample.bin", dtype=np.complex64)
    #print(np.max(in_vector))
    DATA_SIZE = len(in_vector)
    #range_vr = np.max(in_vector)-np.min(in_vector)
    #r2r_threshold = r2r_threshold*range_vr
    #r2r_error = r2r_error*range_vr
    #in_vector = np.zeros((DATA_SIZE,))
    #for i in range(0,int(DATA_SIZE/4)):
    #    in_vector[i] = 0.0
    #for i in range(int(DATA_SIZE/4), int(2*DATA_SIZE/4)):
    #    in_vector[i] = 5.0
    #for i in range(int(2*DATA_SIZE/4), int(3*DATA_SIZE/4)):
    #    in_vector[i] = random.uniform(MIN_D, MAX_D)
    #for i in range(int(3*DATA_SIZE/4), int(3*DATA_SIZE/4)+6):
    #    in_vector[i] = -7.0
    #for i in range(int(3*DATA_SIZE/4)+6, DATA_SIZE):
    #    in_vector[i] = 0.001

    print(DATA_SIZE)
    #in_vector = in_vector.astype('float32')
    in_vector_gpu = cp.asarray(in_vector)
    
    # variable = ctypes.c_size_t(0)
    # outSize = ctypes.pointer(variable)
    for i in range(200):
        s_time = time.time()
        o_bytes, outSize = cuszx_device_compress(in_vector_gpu, r2r_error, DATA_SIZE, 256, r2r_threshold)
        print("Time python: "+str(time.time()-s_time))
        print(outSize[0])
        print("Compress Success...starting decompress ")
        comp = Comp()

        s_time = time.time()
        (d_bytes,ptr )= cuszx_device_decompress(DATA_SIZE*2, o_bytes, comp, in_vector_gpu.dtype)
        
        free_compressed(o_bytes[0])
        cp.cuda.runtime.free(ptr)
        print("Time python: "+str(time.time()-s_time))
    #for i in d_bytes:
    #    print(i)
        print("Decompress Success")
