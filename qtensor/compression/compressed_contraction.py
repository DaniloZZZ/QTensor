import numpy as np

from qtensor.compression import CompressedTensor
from .CompressedTensor import Tensor, iterate_indices
from .CompressedTensor import Compressor

# taken from numpy/core/einsumfunc.py
einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
einsum_symbols_set = set(einsum_symbols)

def contract_two_tensors(A, B, T_out, einsum=np.einsum):
    """
    Contract tensors A and B along their common indices and write result to T_out.
    T_out tensor should be pre-allocated with data.

    This takes care of the case where indices of A and B are Vars with large integer id
    """
    result_indices = T_out.indices
    out_buffer = T_out.data
    max_id = 0
    A_ints = []
    B_ints = []

    for a_i in A.indices:
        a_int = int(a_i)
        max_id = max(max_id, a_int)
        A_ints.append(a_int)

    for b_i in B.indices:
        b_int = int(b_i)
        max_id = max(max_id, b_int)
        B_ints.append(b_int)

    if max_id >= len(einsum_symbols):
        # -- relabel indices to small ints
        all_indices = set(A_ints + B_ints)
        relabel_dict_int = {i: j for j, i in enumerate(all_indices)}
        A_ints = [relabel_dict_int[i] for i in A_ints]
        B_ints = [relabel_dict_int[i] for i in B_ints]
        result_ints = [relabel_dict_int[int(i)] for i in result_indices]
    else:
        result_ints = list(map(int, result_indices))
    #print(A.data.shape)
    #print(B.data.shape)
    out = einsum(A.data, A_ints, B.data, B_ints, result_ints)
    if len(result_ints)>0:
        # This copying is reqiured because cupy doesn't support `out` argument.
        out_buffer[:] = out
    else:
        out_buffer.fill(out)


def compressed_contract(A:Tensor, B: Tensor,
                        contract_ixs, mem_limit,
                        compressor:Compressor,
                        # These two functions are used to support many backends
                        einsum=np.einsum,
                        move_data=lambda x: x
                       ):
    """
    Contract tensors A and B along `contract_ixs` and return the result

    The result tensor indices will be ordered from largest to smallest
    """
    all_indices = list(set(A.indices).union(B.indices))
    all_indices.sort(key=int, reverse=True)
    result_indices = list(set(all_indices) - set(contract_ixs))
    result_indices.sort(key=int, reverse=True)

    # -- Find set of existing compressed that will be decompressed
    exist_compressed = set()
    for T in [A, B]:
        if isinstance(T, CompressedTensor):
            exist_compressed.update(T.slice_indices)
    # In this particular case, we need not to sort these indices,
    # since the iteration over fast index gives same latency as over slow index
    # Potential improvement: if A_S and B_S are different, run outer loop 
    # over min(A_S, B_S) and inner over the rest indices. This will reduce 
    # the number of decompressions.
    # --


    need_compressed = result_indices[:-mem_limit]
    if len(all_indices) == 0:
        new_tensor_name = 'C_s'
        print(f"[Warn]: Scalar inputs in compressed contract, {A}*{B}")
    else:
        new_tensor_name = 'C'+str(int(all_indices[-1]))

    # -- Early return: if no need to compress, do the regular contraction
    if len(need_compressed)==0 and len(exist_compressed)==0:
        C = Tensor.empty(new_tensor_name, result_indices)
        C.data = move_data(C.data)
        contract_two_tensors(A, B, C, einsum=einsum)
        return C
    # --
    print(f"Need compression: {need_compressed}")

    remove_compress = exist_compressed - set(need_compressed)
    R = CompressedTensor(new_tensor_name,
                         result_indices,
                         slice_indices=need_compressed,
                         compressor=compressor
                        )

    result_chunk_ixs = result_indices[-mem_limit:]
    print(f"Chunk indices: {result_chunk_ixs}, remove_compress: {remove_compress}")
    slice_dict = {}
    chunk = np.empty(2**len(result_chunk_ixs), dtype=B.dtype)
    chunk = chunk.reshape(*(v.size for v in result_chunk_ixs))
    chunk = move_data(chunk)
    for r_i in iterate_indices(need_compressed):
        for ix, sl in zip(need_compressed, r_i):
            slice_dict[ix] = sl
        for irm in iterate_indices(remove_compress):
            for i, ival in zip(remove_compress, irm):
                slice_dict[i] = ival#slice(ival, ival+1)
            chunk_view = chunk[tuple(
                slice_dict.get(i, slice(None)) for i in result_chunk_ixs
            )]
            A_slice = A[slice_dict]
            B_slice = B[slice_dict]

            C_ixs = [v for v in result_chunk_ixs if v not in exist_compressed]
            C = Tensor('tmp', indices=C_ixs, data=chunk_view)
            contract_two_tensors(A_slice, B_slice, C, einsum=einsum)
            # Free temp slices
            #import cupy
            #print("Flags", A_slice.data.flags, B_slice.data.flags, C.data.flags)
            #cupy.cuda.runtime.free(A_slice.data.data.ptr)
            #cupy.cuda.runtime.free(B_slice.data.data.ptr)
            compressor.free_decompressed()
        if len(need_compressed)==0:
            R = Tensor(new_tensor_name, result_indices, data=chunk)
        else:
            R.set_chunk(r_i, chunk)
    print('Return', R)
    return R

def compressed_sum(A:Tensor, sum_ixs,
                   compressor:Compressor,
                   mem_limit,
                   # These two functions are used to support many backends
                   einsum=np.einsum,
                   move_data=lambda x: x
                  ):
    """
    The result tensor indices will be ordered from largest to smallest
    """
    all_indices = list(set(A.indices))
    all_indices.sort(key=int, reverse=True)
    result_indices = list(set(all_indices) - set(sum_ixs))
    result_indices.sort(key=int, reverse=True)

    # -- Find set of existing compressed that will be decompressed
    exist_compressed = set()
    if isinstance(A, CompressedTensor):
        exist_compressed.update(A.slice_indices)
    # In this particular case, we need not to sort these indices,
    # since the iteration over fast index gives same latency as over slow index
    # Potential improvement: if A_S and B_S are different, run outer loop 
    # over min(A_S, B_S) and inner over the rest indices. This will reduce 
    # the number of decompressions.
    # --

    need_compressed = result_indices[:-mem_limit]
    new_tensor_name = 'C'+str(int(all_indices[-1]))

    # -- Early return: if no need to compress, do the regular contraction
    if len(need_compressed)==0 and len(exist_compressed)==0:
        C = Tensor.empty(new_tensor_name, result_indices)
        sum_axes = tuple([A.indices.index(i) for i in sum_ixs])
        C.data = A.data.sum(axis=sum_axes)
        return C
    # --
    print(f"Need compression: {need_compressed}")

    remove_compress = exist_compressed - set(need_compressed)
    R = CompressedTensor(new_tensor_name,
                         result_indices,
                         slice_indices=need_compressed,
                         compressor=compressor
                        )

    result_chunk_ixs = result_indices[-mem_limit:]
    print(f"Chunk indices: {result_chunk_ixs}, remove_compress: {remove_compress}")
    slice_dict = {}
    chunk = np.empty(2**len(result_chunk_ixs), dtype=A.dtype)
    chunk = chunk.reshape(*(v.size for v in result_chunk_ixs))
    chunk = move_data(chunk)
    for r_i in iterate_indices(need_compressed):
        for ix, sl in zip(need_compressed, r_i):
            slice_dict[ix] = sl
        for irm in iterate_indices(remove_compress):
            for i, ival in zip(remove_compress, irm):
                slice_dict[i] = ival#slice(ival, ival+1)
            chunk_view = chunk[tuple(
                slice_dict.get(i, slice(None)) for i in result_chunk_ixs
            )]
            A_slice = A[slice_dict]
            sum_axes = [A_slice.indices.index(i) for i in sum_ixs]

            C_ixs = [v for v in result_chunk_ixs if v not in exist_compressed]
            C = Tensor('tmp', indices=C_ixs, data=chunk_view)
            chunk_view[:] = A_slice.data.sum(axis=tuple(sum_axes))
        if len(need_compressed)==0:
            R = Tensor(new_tensor_name, result_indices, data=chunk)
        else:
            R.set_chunk(r_i, chunk)
        compressor.compressor.free_decompressed()
    return R
