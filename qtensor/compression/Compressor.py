import io
import sys
import numpy as np
from pathlib import Path
print(Path(__file__).parent/'szx/src/')
sys.path.append(str(Path(__file__).parent/'szx/src/'))
sys.path.append('./szx/src')
# sys.path.append(str(Path(__file__).parent/'szp/src/'))
# sys.path.append('./szp/src')

sys.path.append(str(Path(__file__).parent/'cusz/src'))
sys.path.append('./cusz/src')
sys.path.append(str(Path(__file__).parent/'torch_quant'))
sys.path.append('./torch_quant')
sys.path.append(str(Path(__file__).parent/'newsz'))
sys.path.append('./newsz')


import torch
try:
    from cuszx_wrapper import cuszx_host_compress, cuszx_host_decompress, cuszx_device_compress, cuszx_device_decompress
    # from cuSZp_wrapper import cuszp_device_compress, cuszp_device_decompress
    from cusz_wrapper import cusz_device_compress, cusz_device_decompress
    from torch_quant_perchannel import quant_device_compress, quant_device_decompress
    from newsz_wrapper import newsz_device_compress, newsz_device_decompress
except:
    print("import failed")
    # Silently fail on missing build of cuszx
    pass

CUSZX_BLOCKSIZE = 256

# -- helper functions

def _get_data_info(data):
    import cupy
    if isinstance(data, cupy.ndarray):
        isCuPy = True
    else:
        isCuPy = False
    num_elements = data.size
    # Adapt numele depending on itemsize
    itemsize = data.dtype.itemsize
    num_elements_eff = int(num_elements*itemsize/4) 
    return isCuPy, num_elements_eff

# -- Compressor classes

class Compressor():
    def compress(self, data):
        raise NotImplementedError

    def decompress(self, ptr):
        raise NotImplementedError

    def compress_size(self, ptr):
        return ptr.nbytes

# -- Debugging and profiling

import time
from dataclasses import dataclass, asdict
@dataclass
class CompressMeasure:
    time: float = 0
    size_in: int = 0
    size_out: int = 0
    label: str = ''

    def __str__(self):
        compress_ratio = self.size_in / self.size_out
        return (f'Measure: {self.time:.3f}s, '
                f'{self.size_in/1024**2:.2f}MB -> {self.size_out/1024**2:.2f}MB ({compress_ratio:.3f} in/out ratio)'
        )

class ProfileCompressor(Compressor):
    def __init__(self, compressor:Compressor, trace=True):
        self.trace = trace
        self.compressor = compressor
        self.profile_data = {'compress': [], 'decompress': []}

    def compress(self, data):
        start = time.time()
        ptr = self.compressor.compress(data)
        end = time.time()
        out_size = self.compressor.compress_size(ptr)
        cmeasure = CompressMeasure(end-start, data.nbytes, out_size)
        self.profile_data['compress'].append(cmeasure)
        if self.trace:
            print(f'Compress: {cmeasure}')
        return ptr

    def decompress(self, ptr):
        start = time.time()
        data = self.compressor.decompress(ptr)
        end = time.time()
        in_size = self.compressor.compress_size(ptr)
        dmeasure = CompressMeasure(end-start, in_size, data.nbytes)
        self.profile_data['decompress'].append(dmeasure)
        if self.trace:
            print(f'Decompress: {dmeasure}')
        return data

    def get_profile_data(self):
        return self.profile_data['compress'], self.profile_data['decompress']

    def get_profile_data_json(self):
        compress, decompress = self.get_profile_data()
        return {
            'compress': [asdict(c) for c in compress],
            'decompress': [asdict(c) for c in decompress],
        }

    def get_profile_stats(self):
        compress, decompress = self.get_profile_data()
        compress_time = sum([x.time for x in compress])
        decompress_time = sum([x.time for x in decompress])
        compress_ratios = np.mean([x.size_in/x.size_out for x in compress])
        compress_size = sum([x.size_out for x in compress])
        return compress_time, decompress_time, compress_size, compress_ratios

    def compress_size(self, ptr):
        return self.compressor.compress_size(ptr)
    
    def free_decompressed(self):
        self.compressor.free_decompressed()
    
    def free_compressed(self, ptr):
        self.compressor.free_compressed(ptr)
# --

class NumpyCompressor(Compressor):
    def compress(self, data):
        comp = io.BytesIO()
        np.savez_compressed(comp, data)
        return comp

    def compress_size(self, ptr):
        return ptr.getbuffer().nbytes

    def decompress(self, ptr):
        ptr.seek(0)
        return  np.load(ptr)['arr_0']

class TorchCompressor(Compressor):
    def __init__(self, r2r_error=1e-3, r2r_threshold=1e-3):
        self.r2r_error = r2r_error
        self.r2r_threshold = r2r_threshold
        self.decompressed_own = []

    def free_decompressed(self):
        import cupy
        print("Cleanup", len(self.decompressed_own))
        for x in self.decompressed_own:
            del x
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()
        self.decompressed_own = []

    def free_compressed(self, ptr):
        cmp_bytes, num_elements_eff, shape, dtype, _ = ptr
        del cmp_bytes

    def compress(self, data):
        isCupy, num_elements_eff = _get_data_info(data)
        dtype = data.dtype
        cmp_bytes, outSize_ptr = quant_device_compress(data, num_elements_eff, CUSZX_BLOCKSIZE, self.r2r_threshold)
        return (cmp_bytes, num_elements_eff, data.shape, dtype, outSize_ptr)

        # return (cmp_bytes, num_elements_eff, isCuPy, data.shape, dtype, outSize_ptr.contents.value)

    def compress_size(self, ptr):
        return ptr[4]

    def decompress(self, obj):
        import cupy
        cmp_bytes, num_elements_eff, shape, dtype, cmpsize = obj
        decompressed_ptr = quant_device_decompress(num_elements_eff, cmp_bytes, self, dtype)
        arr_cp = decompressed_ptr[0]

        arr = cupy.reshape(arr_cp, shape)
        self.decompressed_own.append(arr)
        return arr
    

class NEWSZCompressor(Compressor):
    def __init__(self, r2r_error=1e-3, r2r_threshold=1e-3):
        self.r2r_error = r2r_error
        self.r2r_threshold = r2r_threshold
        self.decompressed_own = []

    def free_decompressed(self):
        import cupy
        print("Cleanup", len(self.decompressed_own))
        for x in self.decompressed_own:
            #print(x)
            #if x == None:
            #    continue
            #else:
                #print("CUDA Free", x)
            cupy.cuda.runtime.free(x)
            # del x
            # cupy.get_default_memory_pool().free_all_blocks()
            # cupy.get_default_pinned_memory_pool().free_all_blocks()
        # torch.cuda.empty_cache()
        self.decompressed_own = []

    def free_compressed(self, ptr):
        import ctypes, cupy
        cmp_bytes, num_elements_eff, isCuPy, shape, dtype, _ = ptr
        p_decompressed_ptr = ctypes.addressof(cmp_bytes[0])
        # cast to int64 pointer
        # (effectively converting pointer to pointer to addr to pointer to int64)
        p_decompressed_int= ctypes.cast(p_decompressed_ptr, ctypes.POINTER(ctypes.c_uint64))
        decompressed_int = p_decompressed_int.contents
        cupy.cuda.runtime.free(decompressed_int.value)

    def compress(self, data):
        isCuPy, num_elements_eff = _get_data_info(data)
        dtype = data.dtype
        cmp_bytes, outSize_ptr = self.cuszx_compress(isCuPy, data, num_elements_eff, self.r2r_error, self.r2r_threshold)
        # return (cmpbytes, num_elements_eff, isCuPy, data.shape, dtype, outSize_ptr)

        return (cmp_bytes, num_elements_eff, isCuPy, data.shape, dtype, outSize_ptr.contents.value)

    def compress_size(self, ptr):
        return ptr[5]

    def decompress(self, obj):
        import cupy
        import ctypes
        cmp_bytes, num_elements_eff, isCuPy, shape, dtype, cmpsize = obj
        decompressed_ptr = self.cuszx_decompress(isCuPy, cmp_bytes, cmpsize, num_elements_eff, self, dtype)
        arr_cp = decompressed_ptr[0]
        self.decompressed_own.append(decompressed_ptr[1])
        
        # -- Workaround to convert GPU pointer to int
        # p_decompressed_ptr = ctypes.addressof(decompressed_ptr)
        # # cast to int64 pointer
        # # (effectively converting pointer to pointer to addr to pointer to int64)
        # p_decompressed_int= ctypes.cast(p_decompressed_ptr, ctypes.POINTER(ctypes.c_uint64))
        # decompressed_int = p_decompressed_int.contents
        # # --
        # self.decompressed_own.append(decompressed_int.value)
        # mem = cupy.cuda.UnownedMemory(decompressed_int.value, num_elements_eff, self, device_id=0)
        # mem_ptr = cupy.cuda.memory.MemoryPointer(mem, 0)
        arr = cupy.reshape(arr_cp, shape)
        # self.decompressed_own.append(arr)
        # arr = cupy.ndarray(shape, dtype=dtype, memptr=mem_ptr)
        return arr
    
    ### Compression API with cuSZx ###
    # Parameters:
    # - isCuPy = boolean, true if data is CuPy array, otherwise is numpy array
    # - data = Numpy or Cupy ndarray, assumed to be 1-D, np.float32 type
    # - num_elements = Number of floating point elements in data
    # - r2r_error = relative-to-value-range error bound for lossy compression
    # - r2r_threshold = relative-to-value-range threshold to floor values to zero
    # Returns:
    # - cmp_bytes = Unsigned char pointer to compressed bytes
    # - outSize_ptr = Pointer to size_t representing length in bytes of cmp_bytes
    def cuszx_compress(self, isCuPy, data, num_elements, r2r_error, r2r_threshold):
        
        if not isCuPy:
            cmp_bytes, outSize_ptr = cuszx_host_compress(data, r2r_error, num_elements, CUSZX_BLOCKSIZE, r2r_threshold)
        else:
            print('Before compress')
            cmp_bytes, outSize_ptr = newsz_device_compress(data, num_elements, CUSZX_BLOCKSIZE, r2r_threshold)
            print('After compress')
            del data
            torch.cuda.empty_cache()
        return cmp_bytes, outSize_ptr

    ### Decompression API with cuSZx ###
    # Parameters:
    # - isCuPy = boolean, true if data is CuPy array, otherwise is numpy array
    # - cmp_bytes = Unsigned char pointer to compressed bytes
    # - num_elements = Number of floating point elements in original data
    # Returns:
    # - decompressed_data = Float32 pointer to decompressed data
    #
    # Notes: Use ctypes to cast decompressed data to Numpy or CuPy type

    def cuszx_decompress(self, isCuPy, cmp_bytes, cmpsize, num_elements, owner, dtype):
        if not isCuPy:
            decompressed_data = cuszx_host_decompress(num_elements, cmp_bytes)
        else:
            # cuszx_device_decompress(nbEle, cmpBytes, owner, dtype)
            decompressed_data = newsz_device_decompress(num_elements, cmp_bytes, owner,dtype)
# oriData, absErrBound, nbEle, blockSize,threshold
            # decompressed_data = quant_device_decompress(num_elements, cmp_bytes, owner,dtype)
        return decompressed_data

class CUSZXCompressor(Compressor):
    def __init__(self, r2r_error=1e-3, r2r_threshold=1e-3):
        self.r2r_error = r2r_error
        self.r2r_threshold = r2r_threshold
        self.decompressed_own = []

    def free_decompressed(self):
        import cupy
        print("Cleanup", len(self.decompressed_own))
        for x in self.decompressed_own:
            #print(x)
            #if x == None:
            #    continue
            #else:
                #print("CUDA Free", x)
            cupy.cuda.runtime.free(x)
            # del x
            cupy.get_default_memory_pool().free_all_blocks()
            #cupy.get_default_pinned_memory_pool().free_all_blocks()
        #torch.cuda.empty_cache()
        self.decompressed_own = []

    def free_compressed(self, ptr):
        import ctypes, cupy
        cmp_bytes, num_elements_eff, isCuPy, shape, dtype, _ = ptr
        p_decompressed_ptr = ctypes.addressof(cmp_bytes[0])
        # cast to int64 pointer
        # (effectively converting pointer to pointer to addr to pointer to int64)
        p_decompressed_int= ctypes.cast(p_decompressed_ptr, ctypes.POINTER(ctypes.c_uint64))
        decompressed_int = p_decompressed_int.contents
        cupy.cuda.runtime.free(decompressed_int.value)
        cupy.get_default_memory_pool().free_all_blocks()
        #cupy.get_default_pinned_memory_pool().free_all_blocks()
        #torch.cuda.empty_cache()

    def compress(self, data):
        isCuPy, num_elements_eff = _get_data_info(data)
        dtype = data.dtype
        cmp_bytes, outSize_ptr = self.cuszx_compress(isCuPy, data, num_elements_eff, self.r2r_error, self.r2r_threshold)
        # return (cmp_bytes, num_elements_eff, isCuPy, data.shape, dtype, outSize_ptr)

        return (cmp_bytes, num_elements_eff, isCuPy, data.shape, dtype, outSize_ptr.contents.value)

    def compress_size(self, ptr):
        return ptr[5]

    def decompress(self, obj):
        import cupy
        import ctypes
        cmp_bytes, num_elements_eff, isCuPy, shape, dtype, cmpsize = obj
        decompressed_ptr = self.cuszx_decompress(isCuPy, cmp_bytes, cmpsize, num_elements_eff, self, dtype)
        arr_cp = decompressed_ptr[0]
        self.decompressed_own.append(decompressed_ptr[1])
        
        # -- Workaround to convert GPU pointer to int
        # p_decompressed_ptr = ctypes.addressof(decompressed_ptr)
        # # cast to int64 pointer
        # # (effectively converting pointer to pointer to addr to pointer to int64)
        # p_decompressed_int= ctypes.cast(p_decompressed_ptr, ctypes.POINTER(ctypes.c_uint64))
        # decompressed_int = p_decompressed_int.contents
        # # --
        # self.decompressed_own.append(decompressed_int.value)
        # mem = cupy.cuda.UnownedMemory(decompressed_int.value, num_elements_eff, self, device_id=0)
        # mem_ptr = cupy.cuda.memory.MemoryPointer(mem, 0)
        arr = cupy.reshape(arr_cp, shape)
        # self.decompressed_own.append(arr)
        # arr = cupy.ndarray(shape, dtype=dtype, memptr=mem_ptr)
        return arr
    
    def cuszx_compress(self, isCuPy, data, num_elements, r2r_error, r2r_threshold):
        """
        ## Compression API with cuSZx ###
        Parameters:
         - isCuPy = boolean, true if data is CuPy array, otherwise is numpy array
         - data = Numpy or Cupy ndarray, assumed to be 1-D, np.float32 type
         - num_elements = Number of floating point elements in data
         - r2r_error = relative-to-value-range error bound for lossy compression
         - r2r_threshold = relative-to-value-range threshold to floor values to zero
         Returns:
         - cmp_bytes = Unsigned char pointer to compressed bytes
         - outSize_ptr = Pointer to size_t representing length in bytes of cmp_bytes
         """
        
        if not isCuPy:
            cmp_bytes, outSize_ptr = cuszx_host_compress(data, r2r_error, num_elements, CUSZX_BLOCKSIZE, r2r_threshold)
        else:
            #cmp_bytes, outSize_ptr = cuszp_device_compress(data, r2r_error, num_elements,  r2r_threshold)
            cmp_bytes, outSize_ptr = cuszx_device_compress(data, r2r_error, num_elements, CUSZX_BLOCKSIZE,r2r_threshold)
            # cmp_bytes, outSize_ptr = quant_device_compress(data, num_elements, CUSZX_BLOCKSIZE, r2r_threshold)
            del data
            torch.cuda.empty_cache()
        return cmp_bytes, outSize_ptr


    def cuszx_decompress(self, isCuPy, cmp_bytes, cmpsize, num_elements, owner, dtype):
        """
        ## Decompression API with cuSZx ###
         Parameters:
         - isCuPy = boolean, true if data is CuPy array, otherwise is numpy array
         - cmp_bytes = Unsigned char pointer to compressed bytes
         - num_elements = Number of floating point elements in original data
         Returns:
         - decompressed_data = Float32 pointer to decompressed data
        
         Notes: Use ctypes to cast decompressed data to Numpy or CuPy type
         """
        if not isCuPy:
            decompressed_data = cuszx_host_decompress(num_elements, cmp_bytes)
        else:
            # cuszx_device_decompress(nbEle, cmpBytes, owner, dtype)
            decompressed_data = cuszx_device_decompress(num_elements, cmp_bytes, owner,dtype)
# oriData, absErrBound, nbEle, blockSize,threshold
            # decompressed_data = quant_device_decompress(num_elements, cmp_bytes, owner,dtype)
        return decompressed_data
    
class CUSZCompressor(Compressor):
    def __init__(self, r2r_error=1e-3, r2r_threshold=1e-3):
        self.r2r_error = r2r_error
        self.r2r_threshold = r2r_threshold
        self.decompressed_own = []

    def free_decompressed(self):
        import cupy
        print("Cleanup", len(self.decompressed_own))
        for x in self.decompressed_own:
            cupy.cuda.runtime.free(x)
            # del x
            # cupy.get_default_memory_pool().free_all_blocks()
            # cupy.get_default_pinned_memory_pool().free_all_blocks()
        # torch.cuda.empty_cache()
        self.decompressed_own = []

    def free_compressed(self, ptr):
        import ctypes, cupy
        cmp_bytes, num_elements_eff, isCuPy, shape, dtype, _ = ptr
        p_decompressed_ptr = ctypes.addressof(cmp_bytes[0])
        # cast to int64 pointer
        # (effectively converting pointer to pointer to addr to pointer to int64)
        p_decompressed_int= ctypes.cast(p_decompressed_ptr, ctypes.POINTER(ctypes.c_uint64))
        decompressed_int = p_decompressed_int.contents
        cupy.cuda.runtime.free(decompressed_int.value)

    def compress(self, data):
        isCuPy, num_elements_eff = _get_data_info(data)

        dtype = data.dtype
        cmp_bytes, outSize_ptr = self.cuszx_compress(isCuPy, data, num_elements_eff, self.r2r_error, self.r2r_threshold)
        return (cmp_bytes, num_elements_eff, isCuPy, data.shape, dtype, outSize_ptr.contents.value)

    def compress_size(self, ptr):
        return ptr[5]

    def decompress(self, obj):
        import cupy
        import ctypes
        cmp_bytes, num_elements_eff, isCuPy, shape, dtype, cmpsize = obj
        decompressed_ptr = self.cuszx_decompress(isCuPy, cmp_bytes, cmpsize, num_elements_eff, self, dtype)
        arr_cp = decompressed_ptr[0]
        #self.decompressed_own.append(decompressed_ptr[1])
        
        # -- Workaround to convert GPU pointer to int
        # p_decompressed_ptr = ctypes.addressof(decompressed_ptr)
        # # cast to int64 pointer
        # # (effectively converting pointer to pointer to addr to pointer to int64)
        # p_decompressed_int= ctypes.cast(p_decompressed_ptr, ctypes.POINTER(ctypes.c_uint64))
        # decompressed_int = p_decompressed_int.contents
        # # --
        # self.decompressed_own.append(decompressed_int.value)
        # mem = cupy.cuda.UnownedMemory(decompressed_int.value, num_elements_eff, self, device_id=0)
        # mem_ptr = cupy.cuda.memory.MemoryPointer(mem, 0)
        arr = cupy.reshape(arr_cp, shape)
        self.decompressed_own.append(arr)
        # arr = cupy.ndarray(shape, dtype=dtype, memptr=mem_ptr)
        return arr
    
    ### Compression API with cuSZx ###
    # Parameters:
    # - isCuPy = boolean, true if data is CuPy array, otherwise is numpy array
    # - data = Numpy or Cupy ndarray, assumed to be 1-D, np.float32 type
    # - num_elements = Number of floating point elements in data
    # - r2r_error = relative-to-value-range error bound for lossy compression
    # - r2r_threshold = relative-to-value-range threshold to floor values to zero
    # Returns:
    # - cmp_bytes = Unsigned char pointer to compressed bytes
    # - outSize_ptr = Pointer to size_t representing length in bytes of cmp_bytes
    def cuszx_compress(self, isCuPy, data, num_elements, r2r_error, r2r_threshold):
        
        if not isCuPy:
            cmp_bytes, outSize_ptr = cuszx_host_compress(data, r2r_error, num_elements, CUSZX_BLOCKSIZE, r2r_threshold)
        else:
            cmp_bytes, outSize_ptr = cusz_device_compress(data, r2r_error, num_elements, CUSZX_BLOCKSIZE,r2r_threshold)
            # cmp_bytes, outSize_ptr = quant_device_compress(data, num_elements, CUSZX_BLOCKSIZE, r2r_threshold)
            del data
            torch.cuda.empty_cache()
        return cmp_bytes, outSize_ptr

    ### Decompression API with cuSZx ###
    # Parameters:
    # - isCuPy = boolean, true if data is CuPy array, otherwise is numpy array
    # - cmp_bytes = Unsigned char pointer to compressed bytes
    # - num_elements = Number of floating point elements in original data
    # Returns:
    # - decompressed_data = Float32 pointer to decompressed data
    #
    # Notes: Use ctypes to cast decompressed data to Numpy or CuPy type

    def cuszx_decompress(self, isCuPy, cmp_bytes, cmpsize, num_elements, owner, dtype):
        if not isCuPy:
            decompressed_data = cuszx_host_decompress(num_elements, cmp_bytes)
        else:
            # cuszx_device_decompress(nbEle, cmpBytes, owner, dtype)
            decompressed_data = cusz_device_decompress(num_elements, cmp_bytes, owner,dtype)
# oriData, absErrBound, nbEle, blockSize,threshold
            # decompressed_data = quant_device_decompress(num_elements, cmp_bytes, owner,dtype)
        return decompressed_data

class WriteToDiskCompressor(Compressor):
    def __init__(self, path):
        from pathlib import Path
        Path(path).mkdir(exist_ok=True, parents=True)
        self.path = path
        self.counter = 1
    
    def _gen_filename(self, info):
        dtype, shape, isCupy = info
        k = np.random.randint(0, 100000000)
        s = hex(k)[2:]
        c = str(self.counter)
        self.counter += 1
        return self.path + f'/qtensor_data_c_{c}_{s}_{str(dtype)}.bin'

    def compress(self, data):
        import cupy
        if isinstance(data, cupy.ndarray):
            isCupy=True
        else:
            isCupy=False
        fname = self._gen_filename((data.dtype, data.shape, isCupy))
        data.tofile(fname)
        return (fname, data.dtype, data.shape, isCupy)

    def compress_size(self, ptr):
        return 0.1

    def decompress(self, obj):
        import cupy
        fname, dtype, shape, isCupy = obj
        if isCupy:
            return cupy.fromfile(fname).view(dtype).reshape(shape)
        else:
            return np.fromfile(fname).view(dtype).reshape(shape)

    def free_compressed(self, ptr):
        pass
    def free_decompressed(self):
        pass
