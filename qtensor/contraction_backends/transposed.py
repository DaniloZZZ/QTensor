from qtensor.contraction_backends import ContractionBackend
from functools import reduce
import qtree
import qtree.optimizer as opt
import qtensor
from qtree import np_framework
from qtensor.tools.lazy_import import opt_einsum
import numpy as np
import string

CHARS = string.ascii_lowercase + string.ascii_uppercase

def opt_einsum_info(expr):
    uniq_ids = set(expr) - {',', '-', '>'}
    sizes_dict = {i:2 for i in uniq_ids}
    views = opt_einsum.helpers.build_views(expr, sizes_dict)
    path, info = opt_einsum.contract_path(expr, *views, optimize='auto')
    return path, info

def get_einsum_expr(bucket, all_indices_list, result_indices):
    # converting elements to int will make stuff faster, 
    # but will drop support for char indices
    # all_indices_list = [int(x) for x in all_indices]
    # to_small_int = lambda x: all_indices_list.index(int(x))
    to_small_int = lambda x: all_indices_list.index(x)
    expr = ','.join(
        ''.join(CHARS[to_small_int(i)] for i in t.indices)
        for t in bucket) +\
            '->'+''.join(CHARS[to_small_int(i)] for i in result_indices)
    return expr

class TransposedBackend(ContractionBackend):
    def __init__(self, device='gpu'):
        super().__init__()
        self.device = device
        #self.pbar = tqdm(desc='Buckets', position=2)
        #self.status_bar = tqdm(desc='Current status', position=3, bar_format='{desc}')

    def process_bucket(self, bucket, no_sum=False):
        all_indices = list(qtensor.utils.merge_sets(set(x.indices) for x in bucket))
        all_indices = sorted(all_indices, key=int)
        result_indices = all_indices[1:]
        contract_index = all_indices[0]
        # print('contract_index', contract_index, no_sum)
        res = self.process_bucket_merged([contract_index], bucket, no_sum=no_sum)
        # Convert indices to growing, bucket elimination expects this
        # print("all_indices:", all_indices)
        # print("bucket:", bucket)
        # print("res.indices:", res.indices)
        # print("result_indices", result_indices)
        tdata = self.get_transpose(res.data,*[res.indices.index(x) for x in result_indices])
        res = qtree.optimizer.Tensor(name=res.name, indices=tuple(result_indices), data=tdata)
        return res

    def _expr_to_ixs(self, expr):
        if isinstance(expr, str):
            inp, out = expr.split('->')
        else:
            pass
        ixs = inp.split(',')
        return ixs, out

    def _get_index_sizes(self, *ixs):
        try:
            sizes = [ i.size for i in ixs ]
        except AttributeError:
            sizes = [2] * len(ixs)
        return sizes

    def _get_index_space_size(self, *ixs):
        sizes = self._get_index_sizes(*ixs)
        return reduce(np.multiply, sizes, 1)

    def _to_letters(self, *ixs):
        all_indices = list(qtensor.utils.merge_sets(set(x) for x in ixs))
        mapping = {i:s for i,s in zip(all_indices, CHARS)}
        convert = lambda x: [mapping[i] for i in x]
        return list(map(convert, ixs))

    def get_tncontract():
        raise NotImplementedError

    def pairwise_sum_contract(self, ixa, a, ixb, b, ixout):
        #ixa, ixb, out = self._to_letters(ixa, ixb, ixout)
        out = ixout
        ixs = ixa, ixb
        tensors = a, b

        # \sum_k A_{kfm} * B_{kfn} = C_{fmn}
        common = set(ixs[0]).intersection(set(ixs[1]))
        kix = common - set(out)
        fix = common - kix

        # -- 1-tensor summation
        all_ix = set(ixa+ixb)
        sum_ix = all_ix - set(out)
        a_sum = sum_ix.intersection(set(ixa) - common)
        b_sum = sum_ix.intersection(set(ixb) - common)
        #print('ab', ixa, ixb, 'shape', a.shape, b.shape, 'sizes', [i.size for i in ixa], [i.size for i in ixb])
        #print('all sum', sum_ix, 'a/b_sum', a_sum, b_sum)
        if len(a_sum):
            a = a.sum(axis=tuple(ixa.index(x) for x in a_sum))
            ixa = [x for x in ixa if x not in a_sum]
        if len(b_sum):
            b = b.sum(axis=tuple(ixb.index(x) for x in b_sum))
            ixb = [x for x in ixb if x not in b_sum]
            #print('ixb', ixb)
        tensors = a, b
        ixs = ixa, ixb
        # --


        mix = set(ixs[0]) - common
        nix = set(ixs[1]) - common
        common = list(kix) + list(fix)

        k, f, m, n = [self._get_index_space_size(*ix)
                      for ix in (kix, fix, mix, nix)
                     ]
        #print('kfmn', k, f, m, n)
        a = self.get_transpose(tensors[0], *[
            list(ixs[0]).index(x) for x in common + list(mix)
        ])

        b = self.get_transpose(tensors[1], *[
            list(ixs[1]).index(x) for x in common + list(nix)
        ])

        #print('ixa', ixa, 'ixb', ixb)
        a = a.reshape(k, f, m)
        b = b.reshape(k, f, n)


        tncontract = self.get_tncontract()

        contraction = 'ijk,ijl->jkl'
        if self.backend == 'cutensor':
            a, b, c, mode_a, mode_b, mode_c, desc_a, desc_b, desc_c = self.get_ready(contraction, a, b)
            G = tncontract(1.0, a, desc_a, mode_a, 
                        b, desc_b, mode_b, 0, 
                        c, desc_c, mode_c)
        else:
            G = tncontract(contraction, a, b)

        if len(out)>17:
            print('lg', G.nbytes)
            #print('ax/bx', ixa, ixb, 'out ix', out, 'kfmnix', kix, fix, mix, nix, 'summed', sum_ix)
        if len(out):
            #print('out ix', out, 'kfmnix', kix, fix, mix, nix)
            G = G.reshape(*self._get_index_sizes(*out))
        current_ord_ = list(fix) + list(mix) + list(nix)
        G = self.get_transpose(G, *[current_ord_.index(i) for i in out])

        return G


    def process_bucket_merged(self, ixs, bucket, no_sum=False):
        #-- merge small tensors with large
        bucket = sorted(bucket, key=lambda x: len(x.indices))
        bucket_tweaked_order = []
        for i, t in enumerate(bucket):
            if len(t.indices)<3:
                bucket_tweaked_order.append(t)
            else:
                bucket_tweaked_order += reversed(bucket[i:])
                break
        #bucket = bucket_tweaked_order

        pairs = []

        for i, sm in enumerate(bucket):
            for j, lg in enumerate(bucket[i+1:]):
                if set(sm.indices).issubset(set(lg.indices)):
                    if len(lg.indices):
                        pairs.append((i, i+j+1))
                        break

        for pair in pairs:
            smi, lgi = pair
            sm, lg = bucket[smi], bucket[lgi]
            #print('merging', sm, lg)
            merg = self.pairwise_sum_contract(
                lg.indices, lg.data, sm.indices, sm.data, lg.indices
            )
            mergt = opt.Tensor(lg.name+'t', lg.indices,
                                data=merg)
            bucket[lgi] = mergt

        removed =[sm for sm, lg in pairs]
        #print('removed', removed)
        bucket = [x for i, x in enumerate(bucket) if i not in removed]
        bucket = list(reversed(bucket))
        if len(removed):
            #print('cobucket', bucket, 'removed', removed)
            pass
        #--

        all_indices = set(sum((list(t.indices) for t in bucket), []))
        result_indices = all_indices - set(ixs)
        #-- find good first merge

        vtup = [tuple(t.indices) for t in bucket]
        #print('vtup', vtup)
        tensors, sum_ix = qtensor.utils.largest_merges(
            vtup , ixs
        )
        if len(sum_ix):
            ia = vtup.index(tensors[0])
            pop_ix = [ia]
            if len(tensors)==1:
                tensor = bucket[ia]
                contr_axes = tuple(tensor.indices.index(x) for x in sum_ix)
                cum_data = self.get_sum(tensor.data, axis=contr_axes)
                cum_indices = tuple(x for x in tensor.indices if x not in sum_ix)
            else:
                ib = vtup.index(tensors[1])
                pop_ix.append(ib)
                a, b = bucket[ia], bucket[ib]
                cum_indices = tuple(set(a.indices).union(b.indices) - set(sum_ix))
                cum_data = self.pairwise_sum_contract(
                    a.indices, a.data, b.indices, b.data, cum_indices
                )

        else:
            pop_ix = [0]
            cum_data, cum_indices = bucket[0].data, bucket[0].indices
        # pop the tensors from bucket
        #print('pop ix', pop_ix)
        bucket = [x for i,x in enumerate(bucket) if i not in pop_ix]

        #--

        #print('bucket', [len(t.indices) for t in bucket])
        #print('subsets', [set(t.indices).issubset(all_indices) for t in bucket])
        #print('contracted', len(ixs), ixs)

        for i, tensor in enumerate(bucket[0:-1]):
            next_indices = set(cum_indices).union(tensor.indices)
            #print('next ix size', len(next_indices), len(cum_indices), len(tensor.indices))
            if len(next_indices)>35:
                print('next size:', len(next_indices), 'bucket result', cum_indices, 'merged', ixs)
                print('bucket', bucket)
            #-- contract indices specific to this pair, if any
            # note that it may be better to reorder pairs of tensors
            # sorting by largest set of pair-specific indices
            _seta, _setb = set(cum_indices), set(tensor.indices)
            _intersec = _seta.intersection(_setb)
            _others = set(sum((list(t.indices) for t in bucket[i+2:]),[]))
            specific_indices = _intersec - _others
            #print('specific indices', specific_indices, 'intersection', _intersec)

            # allow to contract only those indices that are not requested in resutlt
            specific_indices = specific_indices.intersection(set(ixs))

            if len(specific_indices):
                #print('[D] Found specific_indices', specific_indices, 'ixs', ixs)
                pass
            #--
            next_indices = next_indices - specific_indices
            if len(next_indices) > 210:
                print('len next indices', len(next_indices), 'specific indices', specific_indices)
                print('next size:', len(next_indices), 'bucket result', result_indices, 'merged', ixs)
                print('bucket', bucket)

            cum_data = self.pairwise_sum_contract(
                cum_indices, cum_data, tensor.indices, tensor.data, next_indices
            )
            cum_indices = tuple(next_indices)

        if len(bucket):
            tensor = bucket[-1]
            result_data = self.pairwise_sum_contract(
                cum_indices, cum_data, tensor.indices, tensor.data, result_indices
            )
        else:
            result_data = cum_data

        tag = str(list(ixs)[0])

        result = opt.Tensor(f'E{tag}', result_indices,
                            data=result_data)
        return result

    def get_sliced_buckets(self, buckets, data_dict, slice_dict):
        backend = self.backend_module
        sliced_buckets = []
        for bucket in buckets:
            sliced_bucket = []
            for tensor in bucket:
                # get data
                # sort tensor dimensions
                transpose_order = self.get_argsort(list(map(int, tensor.indices)))
                data = data_dict[tensor.data_key]
                data = self.prepare(data)
                data = self.get_transpose(data, tuple(transpose_order))
                # transpose indices
                indices_sorted = [tensor.indices[pp]
                                  for pp in transpose_order]

                # slice data
                slice_bounds = []
                for idx in indices_sorted:
                    try:
                        slice_bounds.append(slice_dict[idx])
                    except KeyError:
                        slice_bounds.append(slice(None))

                data = data[tuple(slice_bounds)]

                # update indices
                indices_sliced = [idx.copy(size=size) for idx, size in
                                  zip(indices_sorted, data.shape)]
                indices_sliced = [i for sl, i in zip(slice_bounds, indices_sliced) if not isinstance(sl, int)]
                assert len(data.shape) == len(indices_sliced)

                sliced_bucket.append(
                    tensor.copy(indices=indices_sliced, data=data))
            sliced_buckets.append(sliced_bucket)

        return sliced_buckets
    
    def get_sum(data, axis):
        raise NotImplementedError
    
    def get_transpose(data, *axis):
        raise NotImplementedError
    
    def prepare(data):
        return data

    def get_argsort(*args):
        raise NotImplementedError
    
    def get_result_data(self, result):
        return result.data

