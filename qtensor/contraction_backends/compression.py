from qtensor.contraction_backends import ContractionBackend
from qtensor.compression import Compressor, CompressedTensor, Tensor
from qtensor.compression.compressed_contraction import compressed_contract, compressed_sum
from qtensor.contraction_backends.common import slice_numpy_tensor
from qtree.optimizer import Tensor

class CompressionBackend(ContractionBackend):
    """
    Compression bucket contraction backend.

    This backend "decorates" another backend, by using compression in 
    pairwise contraction. If the result tensor has more than `max_tw` indices,
    it is sliced and the contraction result is compressed before proceeding to
    next slice.
    """
    def __init__(self, backend, compressor:Compressor, max_tw:int):
        """
        Arguments:
            backend: the backend to use for contraction
            compressor: the compressor to use for compression
            max_tw: threshold for triggering compression.

        """
        self.backend = backend
        self.compressor = compressor
        self.max_tw = max_tw

    def _get_backend_specific_fns(self, backend):
        ## Hacky way to extend backends
        if 'cupy' in backend.__class__.__name__.lower():
            import cupy as cp
            return cp.einsum, cp.array
        elif 'torch' in backend.__class__.__name__.lower():
            import torch
            return torch.einsum, torch.tensor
        else:
            import numpy as np
            return np.einsum, lambda x: x

    def process_bucket(self, bucket, no_sum=False):
        """
        Process a bucket.

        This uses `self.backend.process_bucket` in combination with
        compression.compressed_contraction.compressed_contract
        """
        ctr_kw = dict(zip(['einsum', 'move_data'], self._get_backend_specific_fns(self.backend)))
        bucket.sort(key=lambda x: len(x.indices))
        #print("Processing bucket", bucket)
        accum = bucket[0]
        for t in bucket[1:-1]:
            accum = compressed_contract(
                accum, t, [], self.max_tw, self.compressor,
                **ctr_kw
            )
        if len(bucket)>1:
            t = bucket[-1]
            if len(t.indices) == 0:
                print(f"Scalar tensor {t}, {t.data}")
                accum = accum * t
                return accum
            total_ixs = sorted(
                set().union(*[t.indices, accum.indices])
                , key=int, reverse=True
            )
            accum_new = compressed_contract(
                accum, t, [total_ixs[-1]], self.max_tw, self.compressor
                ,**ctr_kw
            )
            # free data
            import cupy
            for t in [accum, t]:
                if isinstance(t, CompressedTensor):
                    t.compressor.free_decompressed()
                    
            accum = accum_new

            return accum
        else:
            if len(accum.indices) < 1:
                return accum
            indices = (accum.indices[-1], )
            res = compressed_sum(accum, indices, self.compressor, self.max_tw,  **ctr_kw)
            if isinstance(accum, CompressedTensor):
                accum.compressor.free_decompressed()
            return res

    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        """
        Slice buckets accounding to `slice_dict`

        This delegates to `self.backend`, assuming that buckets don't have 
        tensors with more than `self.max_tw` indices.
        """
        # Note: to support large tensors (more than `max_tw`), 
        # just iterate through sliced bucket tensors and compress if needed
        return self.backend.get_sliced_buckets(buckets, data_dict, slice_dict)

    def get_result_data(self, result):
        """
        Get result data from `result` tensor.

        This assumes that the result has at most `self.max_tw` indices.
        """
        return self.backend.get_result_data(result)
