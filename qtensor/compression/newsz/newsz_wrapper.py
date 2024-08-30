import numpy as np
import ctypes
from ctypes import *
import random
from qtensor.tools.lazy_import import cupy as cp
import time
import torch

from pathlib import Path
LIB_PATH = str(Path(__file__).parent/'libnewsz_wrapper.so')
 
NVCOMP_PATH = str(Path(__file__).parent/'libnvcomp.so')
#NVCOMP_PATH= './libnvcomp.so'
#LIB_PATH = './libnewsz_wrapper.so'

# unsigned char* newSZ_device_compress(float *oriData, size_t *outSize, size_t nbEle, int blockSize)
def get_device_compress():
    dll_base = ctypes.CDLL(NVCOMP_PATH,mode=ctypes.RTLD_GLOBAL)
    dll = ctypes.CDLL(LIB_PATH, mode=ctypes.RTLD_GLOBAL)
    func = dll.newSZ_device_compress
    func.argtypes = [POINTER(c_float), POINTER(c_size_t), c_size_t, c_int]
    func.restype = POINTER(c_ubyte)
    return func

# float* newSZ_device_decompress(size_t nbEle, unsigned char* cmpBytes, int blocksize, size_t cmpsize)
def get_device_decompress():

    dll_base = ctypes.CDLL(NVCOMP_PATH,mode=ctypes.RTLD_GLOBAL)
    dll = ctypes.CDLL(LIB_PATH, mode=ctypes.RTLD_GLOBAL)
    func = dll.newSZ_device_decompress
    func.argtypes = [c_size_t, POINTER(c_ubyte), c_int, c_size_t]
    func.restype = POINTER(c_float)
    return func


def newsz_device_compress(oriData, nbEle, blockSize,threshold):
    __cuszx_device_compress = get_device_compress()
    ori_nbEle = nbEle
    variable = ctypes.c_size_t(0)
    outSize = ctypes.pointer(variable)

    oriData = oriData.flatten()
    ori_real = oriData.real
    ori_imag = oriData.imag
    oriData = cp.concatenate((ori_real, ori_imag))
    sample = oriData[::2]
    d = cp.amax(oriData) - cp.amin(oriData)
    d = d.get()
    if d.dtype == np.complex64:
        d = d.real
    threshold = threshold*(d)
    truth_values = abs(oriData)<=threshold
    oriData[truth_values] = 0.0
    nbEle = oriData.shape[0]
    

    oriData_p = ctypes.cast(oriData.data.ptr, ctypes.POINTER(c_float))
    # newSZ_device_compress(float *oriData, size_t *outSize, size_t nbEle, int blockSize)
    o_bytes = __cuszx_device_compress(oriData_p, outSize, np.ulonglong(nbEle), np.int32(blockSize))
    #print("testing")
    #print(o_bytes.value)
    return (o_bytes,outSize.contents.value, blockSize), outSize


def newsz_device_decompress(nbEle, cmpBytes, owner, dtype):
    __cuszx_device_decompress=get_device_decompress()
    (cmpBytes, cmpsize, blockSize) = cmpBytes

    nbEle_p = ctypes.c_size_t(nbEle)
    # float* newSZ_device_decompress(size_t nbEle, unsigned char* cmpBytes, int blocksize, size_t cmpsize)
    newData = __cuszx_device_decompress(nbEle_p, cmpBytes, np.int32(blockSize), ctypes.c_size_t(cmpsize))
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
    mem = cp.cuda.UnownedMemory(decompressed_int.value, nbEle*4, owner, device_id=0)
    mem_ptr = cp.cuda.memory.MemoryPointer(mem, 0)
    #print("mem ptr")
    #print(mem_ptr)
    arr = cp.ndarray(shape=(nbEle,), dtype=np.float32, memptr=mem_ptr)
    # res = cp.zeros((nbEle,))
    # ## need to convert newData to cupy
    # cp.place(res,bitmap,arr)

    c_res = cp.zeros(int(nbEle/2), np.complex64)
    c_res.real = arr[0:int(nbEle/2)]
    c_res.imag = arr[int(nbEle/2):]
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
    #print(in_vector[0:16])
    for i in range(200):
        s_time = time.time()
        #o_bytes, outSize = cusz_device_compress(in_vector_gpu, r2r_error, DATA_SIZE, 256, r2r_threshold)

        o_bytes, outSize = newsz_device_compress(in_vector_gpu, DATA_SIZE, 256,r2r_threshold)
        print("Time python: "+str(time.time()-s_time))
        print(outSize[0])
        print("Compress Success...starting decompress ")
        comp = Comp()

        s_time = time.time()
        #(d_bytes,ptr )= cusz_device_decompress(DATA_SIZE*2, o_bytes, comp, in_vector_gpu.dtype)
        
        (d_bytes, ptr) = newsz_device_decompress(DATA_SIZE*2, o_bytes, comp, in_vector_gpu.dtype)
        free_compressed(o_bytes[0])
        cp.cuda.runtime.free(ptr)
        print("Time python: "+str(time.time()-s_time))
    #for i in d_bytes:
    #    print(i)
        print("Decompress Success")
