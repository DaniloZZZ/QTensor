import numpy as np
from dataclasses import dataclass
from qtensor.contraction_backends import ContractionBackend, NumpyBackend
from qtensor.contraction_backends.compression import CompressionBackend, CompressedTensor
from pyrofiler import timing
from qtensor.tools.lazy_import import torch, pandas
import string

# -- memory profiling
from weakref import WeakValueDictionary

class MemProfBackend(ContractionBackend):
    def __init__(self, backend=NumpyBackend(), print=True):
        self.backend = backend
        self.object_store = WeakValueDictionary()
        self.object_keys = []
        self.print = print
        self.mem_history = []

        import nvidia_smi
        nvidia_smi.nvmlInit()
        self.nvsmi_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

    def _print(self, *args, **kwargs):
        if self.print:
            print(*args, **kwargs)

    def _get_nvsmi_mem(self):
        import nvidia_smi
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(self.nvsmi_handle)
        mem = info.used
        return mem

    @property
    def max_mem(self):
        mems = [m['mem'] for m in self.mem_history]
        return max(mems)

    @property
    def nvsmi_max_mem(self):
        mems = [m['nvmem'] for m in self.mem_history]
        return max(mems)
    @property
    def cupy_buffer_max_mem(self):
        mems = [m['cupy_bufsize'] for m in self.mem_history]
        return max(mems)
    @property
    def object_max_mem(self):
        mems = [m['objmem'] for m in self.mem_history]
        return max(mems)

    def check_store(self):
        import cupy
        mempool = cupy.get_default_memory_pool()
        total_mem = 0
        deleted_keys = []
        for key in self.object_keys:
            tensor = self.object_store.get(key, None)
            if tensor is None:
                #self._print("Tensor", key, "was deleted")
                deleted_keys.append(key)
                continue
            else:
                size = self.tensor_size(tensor)
                if isinstance(tensor, CompressedTensor):
                    print("Tensor", tensor, "size", size)
                total_mem += size
        for key in deleted_keys:
            self.object_keys.remove(key)

        if total_mem>1024**2:
            self._print("Total memory usage", total_mem/1024/1024, "MB")
            mempool.free_all_blocks()
        cupy_mem = mempool.used_bytes()
        # get maximum memory usage
        gpu_mem = cupy_mem
        if isinstance(self.backend, CompressionBackend):
            gpu_mem += 8*2**self.backend.max_tw
        self.mem_history.append(dict(
            mem=gpu_mem,
            cupy_bufsize=mempool.total_bytes(),
            nvmem = self._get_nvsmi_mem(),
            cupybuf=mempool.total_bytes(),
            objmem=total_mem,
            tensors_sizes=[len(tensor.indices) for tensor in self.object_store.values()]
        ))
        # --
        #print('MH', self.mem_history[-1])
        if cupy_mem>1024**2:
            self._print("CuPy memory usage", cupy_mem/1024/1024, "MB. Total MB:", mempool.total_bytes()/1024**2)

    def tensor_size(self, tensor)->int:
        from qtensor.compression import Tensor, CompressedTensor
        if tensor.data is None:
            return 0
        if isinstance(tensor, CompressedTensor):
            chunks = tensor._data
            sizes = [tensor.compressor.compress_size(x) for x in chunks]
            return sum(sizes)
        elif isinstance(tensor, Tensor):
            return tensor.data.nbytes
        else:
            raise ValueError("Unknown tensor type")

    def add_tensor(self, tensor):
        label = str(tensor)
        self.object_store[label] = tensor
        self.object_keys.append(label)
        tsize = self.tensor_size(tensor)
        if tsize>1024:
            self._print("Added tensor with data size", tsize/1024, "KB")
        self.check_store()

    def process_bucket(self, bucket, no_sum=False):
        res = self.backend.process_bucket(bucket, no_sum=no_sum)
        self.add_tensor(res)
        return res

    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        buckets = self.backend.get_sliced_buckets(buckets, data_dict, slice_dict)
        for bucket in buckets:
            for tensor in bucket:
                self.add_tensor(tensor)
        return buckets

    def get_result_data(self, result):
        return self.backend.get_result_data(result)

# --

@dataclass
class BucketContnractionStats:
    """
    Time for contraction in seconds
    """
    time: float
    """ tensor indices"""
    indices: list
    """ Strides for each tensor. List of lists of ints"""
    strides: list

    @classmethod
    def from_bucket_time(cls, bucket: list, time: float):
        # Note: do not store bucket data to avoid memory leak
        indices = [t.indices for t in bucket]
        strides = []
        for t in bucket:
            # numpy
            if hasattr(t.data, 'strides'):
                tstr = [s/t.data.itemsize for s in t.data.strides]
                tstr = [int(np.log2(s)) for s in tstr]
                strides.append(tstr)
            # Torch
            elif hasattr(t.data, 'stride'):
                tstr = [int(np.log2(s)) for s in t.data.stride()]
                strides.append(tstr)
            else:
                strides.append([0]*len(t.indices))

        return cls(time, indices, strides)

    def indices_info(self):
        """ String representation of bucket data"""
        info = ""
        all_indices = sorted(list(set(sum(map(list, self.indices), []))), key=int)
        ix_to_char = {i:string.ascii_letters[j] for j, i in enumerate(all_indices)}
        for ix, strides in zip(self.indices, self.strides):
            tensor_info = ""
            for i, s in zip(ix, strides):
                tensor_info += f"{ix_to_char[i]}{s} "
            info += f"({tensor_info})"
        smallest = all_indices[0]
        return f"Σ{ix_to_char[smallest]} {info}"

REPORT_COLUMNS = ('bucket_len', 'time', 'flop', 'FLOPS', 'max_size', 'min_size', 'result_size', 'indices')

class PerfBackend(ContractionBackend):
    Backend = ContractionBackend

    def __init__(self, *args, print=False, num_lines=20, **kwargs):
        self.backend = self.Backend(*args, **kwargs)
        self._print = print
        self.max_lines = num_lines
        self._profile_results = {}
        self.report_table = pandas.DataFrame(columns=REPORT_COLUMNS)

    def _profile_callback(self, time, label, bucket):
        if self._print:
            print(f"PROF:: perf data {label}: {time}")
        self._profile_results[str(id(bucket))] = BucketContnractionStats.from_bucket_time(bucket, time)

    @classmethod
    def from_backend(cls, backend, *args, **kwargs):
        """ Dynamically create and instantiate a class with a given backend. """
        class CustomGeneratedBackend(cls):
            Backend = backend
        return CustomGeneratedBackend(*args, **kwargs)

    def process_bucket(self, bucket, no_sum=False):
        with timing('process bucket time', bucket
                         , callback=self._profile_callback):
            return self.backend.process_bucket(bucket, no_sum=no_sum)

    def process_bucket_merged(self, ixs, bucket, no_sum=False):
        with timing('process bucket time', bucket
                         , callback=self._profile_callback):
            return self.backend.process_bucket_merged(ixs, bucket, no_sum=no_sum)

    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        return self.backend.get_sliced_buckets(buckets, data_dict, slice_dict)

    def get_result_data(self, result):
        return self.backend.get_result_data(result)

    def _pairwise_flop_mem(self, indices, contract_last=0):
        """
        Args:
            indices: list of two index lists
        Returns:
            next_indices: list of indices after contraction
            tuple(flops, mem): required resources for contraction
        """
        next_indices = list(set().union(*indices))
        next_indices.sort(key=int, reverse=True)
        flop = np.prod([2 for i in next_indices])
        if contract_last:
            next_indices = next_indices[:-contract_last]
        mem = 0
        for ilist in [next_indices]+indices:
            mem += np.prod([i.size for i in ilist])
        return next_indices, (flop, mem)

    def _perfect_bucket_flop_mem(self, bucket_indices, show = False):
        """
        Returns estimation of flops for a bucket
        """
        bucket_indices = sorted(bucket_indices, key=lambda x: len(x))
        flop_list = []
        mem_list = []
        accum = bucket_indices[0]
        for ixs in bucket_indices[1:-1]:
            indices = [accum, ixs]
            # -- Get pairwise contraction flops
            accum, (flop, mem) = self._pairwise_flop_mem(indices)
            # --
            flop_list.append(flop)
            mem_list.append(mem)
        # -- last contraction removes the smallest index
        indices = [accum, bucket_indices[-1]]
        _, (flop, mem) = self._pairwise_flop_mem(indices, contract_last=1)
        # --
        flop_list.append(flop)
        mem_list.append(mem)

        return sum(flop_list), max(mem_list)



    def gen_report(self, show = True):
        data = list(self._profile_results.values())
        # -- sotrt data with respect to time
        #data = sorted(data, key= lambda pair: pair[1], reverse=True)
        data_repr = list((x.indices_info(), x.time) for x in data)
        # -- report on largest contractions
        max_lines = self.max_lines

        df = pandas.DataFrame(data_repr, columns=['indices', 'time'])

        df.sort_values(by='time', ascending=False, inplace=True)
        rep = df.head(max_lines).to_string()
        if len(data) > max_lines:
            rep += f'\n ... and {len(data)-max_lines} lines more...'

        # -- report on totals
        # max_line should not be inolved for recording

        report_data = {
            'bucket_len': [],
            'time': [],
            'flop': [],
            'FLOPS': [],
            'max_size': [],
            'min_size': [],
            'result_size': [],
            'indices': [],
        }
        for bucket_cstats in  data:
            indices = bucket_cstats.indices
            time = bucket_cstats.time
            max_size = max([len(i) for i in indices])
            min_size = min([len(i) for i in indices])
            flop, mem = self._perfect_bucket_flop_mem(indices)
            result_size = len(set.union(*[set(i) for i in indices])) - 1
            report_data['bucket_len'].append(len(indices))
            report_data['time'].append(time)
            report_data['flop'].append(flop)
            report_data['FLOPS'].append(flop/time)
            report_data['max_size'].append(max_size)
            report_data['min_size'].append(min_size)
            report_data['result_size'].append(result_size)
            report_data['indices'].append(bucket_cstats.indices_info())

        self.report_table = pandas.DataFrame(report_data)
        if show:
            print(self.report_table.to_markdown())


        # -- report on totals
        total_data = len(data)
        total_time = sum(d.time for d in data)
        rep += '\n======\n'
        rep += 'Total time: ' + str(total_time)
        rep += '\nTotal bucket contractions: ' + str(total_data)
        rep += '\nMean time for contraction: ' + str(total_time/total_data)
        rep += '\n'
        return rep

class PerfNumpyBackend(PerfBackend):
    Backend = NumpyBackend


class GPUPerfBackend(PerfBackend):
    def process_bucket(self, bucket, no_sum=False):
        indices = [tensor.indices for tensor in bucket]

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        out = self.backend.process_bucket(bucket, no_sum=no_sum)
        
        end.record()
        torch.cuda.synchronize()
        time= start.elapsed_time(end)/1000

        # sorted(self.backend.exprs.items(), key=lambda x: x[1], reverse=True)
        # print("summary:",sorted(self.backend.exprs.items(), key=lambda x: x[1], reverse=True))

        self._profile_callback(time,'process bucket time',indices)
        return out
