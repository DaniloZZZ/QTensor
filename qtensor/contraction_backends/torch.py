from qtensor.tools.lazy_import import torch
import qtree
import numpy as np
from functools import reduce
from qtree import np_framework
from qtensor.contraction_backends import ContractionBackend
from .common import get_slice_bounds, get_einsum_expr, slice_numpy_tensor
import string
from loguru import logger

CHARS = string.ascii_lowercase + string.ascii_uppercase


def qtree2torch_tensor(tensor, data_dict):
    """Converts qtree tensor to pytorch tensor using data dict"""
    if isinstance(tensor.data, torch.Tensor):
        return tensor
    if tensor.data is not None:
        data = tensor.data
    else:
        data = data_dict[tensor.data_key]
    torch_t = torch.from_numpy(data)
    data_dict[tensor.data_key] = torch_t
    return tensor.copy(data=torch_t)


def get_einsum_expr_bucket(bucket, all_indices_list, result_indices):
    # converting elements to int will make stuff faster,
    # but will drop support for char indices
    # all_indices_list = [int(x) for x in all_indices]
    # to_small_int = lambda x: all_indices_list.index(int(x))
    to_small_int = lambda x: all_indices_list.index(x)
    expr = (
        ",".join("".join(CHARS[to_small_int(i)] for i in t.indices) for t in bucket)
        + "->"
        + "".join(CHARS[to_small_int(i)] for i in result_indices)
    )
    return expr


def permute_torch_tensor_data(data: np.ndarray, indices_in, indices_out):
    """
    Permute the data of a numpy tensor to the given indices_out.

    Returns:
        permuted data
    """
    # permute indices
    in_locs = {idx: i for i, idx in enumerate(indices_in)}
    perm = [in_locs[i] for i in indices_out]
    # permute tensor
    return torch.permute(data, perm)


def slice_torch_tensor(data: np.ndarray, indices_in, indices_out, slice_dict):
    """
    Args:
        data : np.ndarray
        indices_in: list of `qtree.optimizer.Var`
        indices_out: list of `qtree.optimizer.Var`
        slice_dict: dict of `qtree.optimizer.Var` to `slice`

    Returns:
        new data, new indices
    """
    slice_bounds = get_slice_bounds(slice_dict, indices_in)
    s_data = data[slice_bounds]
    indices_sliced = [
        i for sl, i in zip(slice_bounds, indices_in) if not isinstance(sl, int)
    ]
    #print(f'{indices_in=}, {indices_sliced=} {slice_dict=}, {slice_bounds=}, slicedix {indices_sliced}, sshape {s_data.shape}')
    indices_out = [
        v for v in indices_out if not isinstance(slice_dict.get(v, None), int)
    ]
    assert len(indices_sliced) == len(s_data.shape)
    st_data = permute_torch_tensor_data(s_data, indices_sliced, indices_out)
    indices_sized = [v.copy(size=size) for v, size in zip(indices_out, st_data.shape)]
    assert len(indices_sized) == len(st_data.shape)
    return st_data, indices_sized


class TorchBackend(ContractionBackend):
    def __init__(self, device="cpu"):
        # alias of gpu -> cuda
        if device == "gpu":
            device = "cuda"
        # Check that CUDA is available if specified
        if device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("Cuda is not available. Falling back to CPU")
                device = "cpu"
        if device == "xpu":
            import intel_extension_for_pytorch as ipex

        self.device = torch.device(device)
        logger.debug("Torch backend using device {}", self.device)
        self.dtype = ["float", "double", "complex64", "complex128"]
        self.width_dict = [set() for i in range(30)]
        self.width_bc = [[0, 0] for i in range(30)]  # (#distinct_bc, #bc)

    def process_bucket(self, bucket, no_sum=False):
        bucket.sort(key=lambda x: len(x.indices))
        result_indices = bucket[0].indices
        result_data = bucket[0].data
        width = len(set(bucket[0].indices))

        for tensor in bucket[1:-1]:
            expr = get_einsum_expr(
                list(map(int, result_indices)), list(map(int, tensor.indices))
            )

            logger.trace(
                "Before contract. Expr: {}, inputs: {}, {}", expr, result_data, tensor
            )
            result_data = torch.einsum(expr, result_data, tensor.data)
            logger.trace(
                "expression {}. Data: {}, -> {}", expr, tensor.data, result_data
            )

            # Merge and sort indices and shapes
            result_indices = tuple(
                sorted(set(result_indices + tensor.indices), key=int, reverse=True)
            )

            size = len(set(tensor.indices))
            if size > width:
                width = size

            self.width_dict[width].add(expr)
            self.width_bc[width][0] = len(self.width_dict[width])
            self.width_bc[width][1] += 1

        if len(bucket) > 1:
            tensor = bucket[-1]
            expr = get_einsum_expr(
                list(map(int, result_indices)),
                list(map(int, tensor.indices)),
                contract=0 if no_sum else 1,
            )
            logger.trace(
                "Before contract. Expr: {}, inputs: {}, {}", expr, result_data, tensor.data
            )
            result_data = torch.einsum(expr, result_data, tensor.data)
            logger.trace(
                "expression {}. Data: {}, -> {}", expr, tensor.data, result_data
            )
            result_indices = tuple(
                sorted(set(result_indices + tensor.indices), key=int, reverse=True)
            )
        else:
            if not no_sum:
                result_data = result_data.sum(axis=-1)
            else:
                result_data = result_data

        if len(result_indices) > 0:
            first_index = result_indices[-1]
            if not no_sum:
                result_indices = result_indices[:-1]
            tag = first_index.identity
        else:
            tag = "f"
            result_indices = []

        # reduce
        result = qtree.optimizer.Tensor(f"E{tag}", result_indices, data=result_data)
       # print("returning result", [t.data.sum() for t in bucket], bucket, result.data.sum(), no_sum)
        return result

    def process_bucket_merged(self, ixs, bucket, no_sum=False):
        result_indices = bucket[0].indices
        # print("result_indices", result_indices)
        result_data = bucket[0].data
        all_indices = set(sum((list(t.indices) for t in bucket), []))
        result_indices = all_indices - set(ixs)
        all_indices_list = list(all_indices)
        to_small_int = lambda x: all_indices_list.index(x)
        tensors = []
        is128 = False
        for tensor in bucket:
            if tensor.data.dtype in [torch.float64]:
                tensors.append(tensor.data.type(torch.complex64))
            else:
                tensors.append(tensor.data)
            if tensor.data.dtype == torch.complex128:
                is128 = True

        if is128:
            for i in range(len(tensors)):
                tensors[i] = tensors[i].type(torch.complex128)

        expr = get_einsum_expr_bucket(bucket, all_indices_list, result_indices)
        expect = len(result_indices)
        result_data = torch.einsum(expr, *tensors)

        if len(result_indices) > 0:
            first_index, *_ = result_indices
            tag = str(first_index)
        else:
            tag = "f"

        result = qtree.optimizer.Tensor(f"E{tag}", result_indices, data=result_data)

        return result

    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        sliced_buckets = []
        for bucket in buckets:
            sliced_bucket = []
            for tensor in bucket:
                # get data
                # sort tensor dimensions
                out_indices = list(sorted(tensor.indices, key=int, reverse=True))
                if tensor.data is None:
                    data = data_dict[tensor.data_key]
                else:
                    data = tensor.data
                #print("data", data.shape, tensor.data_key, data.sum(), tensor.data is None)
                # Works for torch tensors just fine
                if not isinstance(data, torch.Tensor):
                    data = torch.from_numpy(data.astype(np.complex128)).to(self.device)
                else:
                    data = data.type(torch.complex128)
                # slice data
                data, new_indices = slice_torch_tensor(
                    data, tensor.indices, out_indices, slice_dict
                )
                #print("slice_dict", slice_dict)
                #print("tensor", tensor)
                #print("tensorshape", data.shape)
                #print("tensorindices", new_indices)
                #print("tensorindicessizes", [v.size for v in new_indices])

                #print("sliced tensor: T, daata, indices", tensor, data.sum(), new_indices, ', old:', tensor.indices)
                sliced_bucket.append(tensor.copy(indices=new_indices, data=data))
            sliced_buckets.append(sliced_bucket)

        return sliced_buckets

    def get_result_data(self, result):
        return torch.permute(result.data, tuple(reversed(range(result.data.ndim))))


def _swap_flattened(data, a: int, b: int, sprod, different_dims=False):
    """
    Swap two dimensions in a flattened tensor.

    Args:
        data: flattened tensor
        a, b: dimensions to swap
        sprod (iterable of ints): ith element is the product of dimensions 0 to i Last element should be 1
    """
    if a == b:
        return data
    ndim = len(sprod) - 1
    assert ndim >= max(a, b)
    a, b = min(a, b), max(a, b)
    d5 = data.reshape(
        (
            sprod[a - 1],
            sprod[a] // sprod[a - 1],
            sprod[b - 1] // sprod[a],
            sprod[b] // sprod[b - 1],
            sprod[ndim - 1] // sprod[b],
        )
    )
    # -- modify sprod accordingly
    if different_dims:
        adim = sprod[a] // sprod[a - 1]
        bdim = sprod[b] // sprod[b - 1]
        for i in range(a, b):
            sprod[i] *= bdim
            sprod[i] //= adim
    return d5.transpose(1, 3).flatten()


def permute_flattened(data, perm, shape):
    """
    Permute the data of a many-dimensional tensor stored as a flattened array.
    This is a workaround for the limitation of 12 dimensions in intel extension
    for pytorch.

    While permuting, tensor is reshaped to maximum of 5 dimensions:

    for each dimension swap a-b:
        1. Reshape to 5-dimensional tensor ... a ... b ...
        2. Swap a and b.
        3. Flatten to 1-dimensional tensor.

    Args:
        data: flattened data
        perm (iterable of ints): permutation, as in torch.permute
        shape (iterable of ints): shape of the original tensor

    Returns:
        permuted data, equivalent to torch.permute(data.reshape(shape), perm).flatten()
    """
    sprod = []
    k = 1
    different_dims = False
    for i in shape:
        if i != shape[0]:
            different_dims = True
        k *= i
        sprod.append(k)
    sprod.append(1)
    # print(f'different_dims {different_dims}')
    # Is there a way to use only one dict?
    d2l = {i: i for i in range(len(shape))}

    l2d = {i: i for i in range(len(shape))}
    for t, s in enumerate(perm):
        s = d2l[s]
        data = _swap_flattened(data, s, t, sprod, different_dims)
        l2d[s], l2d[t] = l2d[t], l2d[s]
        d2l[l2d[s]], d2l[l2d[t]] = s, t
        # print(f'{s=}, {t=}, {d2l=}, {l2d=}')
    return data

def sum_flattened(data, axes, shape):
    sprod = []
    k = 1
    for i in shape:
        k *= i
        sprod.append(k)
    sprod.append(1)
    ndim = len(shape)
    for ix in axes:
        ixd = sprod[ix] // sprod[ix - 1]
        d3 = data.reshape(
            (
                sprod[ix - 1],
                ixd,
                sprod[ndim-1] // sprod[ix],
            )
        )
        sprod = sprod[:ix] + list(np.array(sprod[ix + 1:]) * ixd)
        data = d3.sum(axis=1).flatten()
    return data



class TorchBackendMatm(TorchBackend):
    def _get_index_sizes(self, *ixs, size_dict=None):
        if size_dict is not None:
            return [size_dict[i] for i in ixs]
        try:
            sizes = [i.size for i in ixs]
        except AttributeError:
            sizes = [2] * len(ixs)
        return sizes

    def _get_index_space_size(self, *ixs, size_dict=None):
        sizes = self._get_index_sizes(*ixs, size_dict=size_dict)
        return reduce(np.multiply, sizes, 1)

    def pairwise_sum_contract(self, ixa, a, ixb, b, ixout, size_dict=None):
        out = ixout
        common = set(ixa).intersection(set(ixb))
        # -- sum indices that are in one tensor only
        all_ix = set(ixa + ixb)
        sum_ix = all_ix - set(out)
        a_sum = sum_ix.intersection(set(ixa) - common)
        b_sum = sum_ix.intersection(set(ixb) - common)
        # print('ab', ixa, ixb)
        # print('all sum', sum_ix, 'a/b_sum', a_sum, b_sum)
        if len(a_sum):
            #a = a.sum(axis=tuple(ixa.index(x) for x in a_sum))
            a = sum_flattened(a, [ixa.index(x) for x in a_sum], self._get_index_sizes(*ixa, size_dict=size_dict))
            ixa = [x for x in ixa if x not in a_sum]

        if len(b_sum):
            #b = b.sum(axis=tuple(ixb.index(x) for x in b_sum))
            b = sum_flattened(b, [ixb.index(x) for x in b_sum], self._get_index_sizes(*ixb, size_dict=size_dict))
            ixb = [x for x in ixb if x not in b_sum]
        tensors = a, b
        # --

        ixs = ixa, ixb
        common = set(ixs[0]).intersection(set(ixs[1]))

        # \sum_k A_{kfm} * B_{kfn} = C_{fmn}
        mix = set(ixs[0]) - common
        nix = set(ixs[1]) - common
        kix = common - set(out)
        fix = common - kix
        common = list(kix) + list(fix)
        # print(f'{ixa=} {ixb=} {ixout=}; {common=} {mix=} {nix=}, {size_dict=}')
        if tensors[0].numel() > 1:
            # a = tensors[0].permute(*[
            #    list(ixs[0]).index(x) for x in common + list(mix)
            # ])
            a = permute_flattened(
                tensors[0],
                [list(ixs[0]).index(x) for x in common + list(mix)],
                self._get_index_sizes(*ixa, size_dict=size_dict),
            )

        if tensors[1].numel() > 1:
            # b = tensors[1].permute(*[
            #    list(ixs[1]).index(x) for x in common + list(nix)
            # ])
            b = permute_flattened(
                tensors[1],
                [list(ixs[1]).index(x) for x in common + list(nix)],
                self._get_index_sizes(*ixb, size_dict=size_dict),
            )

        k, f, m, n = [
            self._get_index_space_size(*ix, size_dict=size_dict)
            for ix in (kix, fix, mix, nix)
        ]
        a = a.reshape(k, f, m)
        b = b.reshape(k, f, n)
        c = torch.einsum("kfm, kfn -> fmn", a, b)
        #if len(out):
            # print('out ix', out, 'kfmnix', kix, fix, mix, nix)
            #c = c.reshape(*self._get_index_sizes(*out, size_dict=size_dict))
        #print('outix', out, 'res', c.shape, 'kfmn',kix, fix, mix, nix)

        current_ord_ = list(fix) + list(mix) + list(nix)
        c = c.flatten()
        if len(out):
             #c = c.permute(*[current_ord_.index(i) for i in out])
            c = permute_flattened(
                c,
                [current_ord_.index(i) for i in out],
                self._get_index_sizes(*out, size_dict=size_dict),
            )
        # print(f'c shape {c.shape}')
        return c

    def process_bucket(self, bucket, no_sum=False):
        bucket.sort(key=lambda x: len(x.indices))
        result_indices = bucket[0].indices
        result_data = bucket[0].data
        width = len(set(bucket[0].indices))
        #print("bucket", bucket)

        for tensor in bucket[1:-1]:
            ixr = list(map(int, result_indices))
            ixt = list(map(int, tensor.indices))
            out_indices = tuple(
                sorted(set(result_indices + tensor.indices), key=int, reverse=True)
            )
            ixout = list(map(int, out_indices))

            logger.trace("Before contract. expr: {}, {} -> {}", ixr, ixt, ixout)
            size_dict = {}
            for i in result_indices:
                size_dict[int(i)] = i.size
            for i in tensor.indices:
                size_dict[int(i)] = i.size
            logger.trace("result_indices: {}, out_indices {}, tensor {}, tensor.data.shape {}", result_indices, out_indices, tensor, tensor.data.shape)
            result_data_new = self.pairwise_sum_contract(
                ixr, result_data, ixt, tensor.data, ixout, size_dict=size_dict
            )
            result_indices = out_indices
            # result_data = torch.einsum(expr, result_data, tensor.data)
            logger.trace(
                "Data: {}, {} -> {}",
                result_data.shape,
                tensor.data.shape,
                result_data_new.shape,
            )
            result_data = result_data_new

            # Merge and sort indices and shapes

            size = len(set(tensor.indices))
            if size > width:
                width = size

        if len(bucket) > 1:
            tensor = bucket[-1]

            ixr = list(map(int, result_indices))
            ixt = list(map(int, tensor.indices))
            out_indices = tuple(
                sorted(set(result_indices + tensor.indices), key=int, reverse=True)
            )
            if not no_sum:
                out_indices = out_indices[:-1]
            ixout = list(map(int, out_indices))

            logger.trace("Before contract. expr: {}, {} -> {}", ixr, ixt, ixout)
            size_dict = {}
            for i in result_indices:
                size_dict[int(i)] = i.size
            for i in tensor.indices:
                size_dict[int(i)] = i.size
            logger.trace("result_indices: {}, out_indices {}, tensor {}, tensor.data.shape {}", result_indices, out_indices, tensor, tensor.data.shape)
            result_data_new = self.pairwise_sum_contract(
                ixr, result_data, ixt, tensor.data, ixout, size_dict=size_dict
            )
            result_indices = out_indices
            # result_data = torch.einsum(expr, result_data, tensor.data)
            logger.trace(
                "Data: {}, {} -> {}",
                result_data.mean(),
                tensor.data.mean(),
                result_data_new.mean(),
            )
            #print("result_data", result_data_new.shape)
            #print("result_indices", result_indices)
            #print("ixonut", ixout)
            #print("result_indicessizes", [v.size for v in result_indices])
            #print("size_dict", size_dict)
            # if result_data_new.mean() == 0:
            #    logger.warning("Result is zero")
            #    logger.debug("result_indices: {}", result_indices)
            #    logger.debug("result_data: {}", result_data)
            #    logger.debug("tensor: {}", tensor)
            #    logger.debug("tensor_data: {}", tensor.data)
            #    logger.debug("result_data_new: {}", result_data_new)
            #    raise ValueError("Result is zero")
            result_data = result_data_new
        else:
            # Sum the last index
            #print("result_data", result_data.shape)
            #print("result_indices", result_indices)
            #print("result_indicessizes", [v.size for v in result_indices])
            shape = self._get_index_sizes(*result_indices)
            #print("shape", shape)
            #print("no_sum", no_sum)
            if not no_sum:
                #print("reshaping",(-1, shape[-1]))
                result_data = result_data.reshape(-1, shape[-1]).sum(axis=-1)
            #result_data = result_data.sum(axis=-1)
                result_indices = result_indices[:-1]

        if len(result_indices) > 0:
            first_index = result_indices[-1]
            tag = first_index.identity
        else:
            tag = "f"
            result_indices = []

        # reduce
        result = qtree.optimizer.Tensor(f"E{tag}", result_indices, data=result_data)
        #print("returning result", result)
        #print("returning result_data.shape", result_data.shape)
        #print("returning result", [t.data.sum() for t in bucket], bucket,'r', result, result.data.sum(), no_sum)
        #print(f'{result.name}({len(result.indices)})', end='', flush=True)
        return result


    def get_result_data(self, result):
        # In theory, This condition is redundant, both should be either True or False.
        if len(result.indices) or result.data.ndim > 1:
            d = result.data.reshape(self._get_index_sizes(*result.indices))
        else:
            d = result.data
        # move to cpu
        d = d.cpu()
        return torch.permute(d, tuple(reversed(range(d.ndim))))
