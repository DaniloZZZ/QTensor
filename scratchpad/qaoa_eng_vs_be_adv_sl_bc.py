import json
from functools import lru_cache
import numpy as np
import networkx as nx
import platform
import pyrofiler
import qtensor
from qtensor import QtreeQAOAComposer
from qtensor import QAOAQtreeSimulator
from qtensor.contraction_backends import get_backend, get_cpu_perf_backend, get_gpu_perf_backend

gpu_backends = ['torch_gpu', 'cupy', 'tr_torch', 'tr_cupy', 'tr_cutensor']
cpu_backends = ['einsum', 'torch_cpu', 'mkl', 'opt_einsum', 'tr_einsum', 'opt_einsum']

timing = pyrofiler.timing


@lru_cache
def get_test_problem(n=10, p=2, d=3, type='random'):
    #print('Test problem: n, p, d', n, p, d)
    if type == 'random':
        seed = 250
        G = nx.random_regular_graph(d, n, seed = 250)
    elif type == 'grid2d':
        G = nx.grid_2d_graph(n,n)
    elif type == 'line':
        G = nx.Graph()
        G.add_edges_from(zip(range(n-1), range(1, n)))
    gamma, beta = [np.pi/5]*p, [np.pi/2]*p
    return G, gamma, beta

def obj2dict(obj):
    keys = [x for x in dir(obj) if x[0]!='_']
    return dict((key, obj.__getattribute__(key)) for key in keys)


@lru_cache
def get_gpu_props_json():
    try:
        import torch
        devprops = torch.cuda.get_device_properties(torch.cuda.current_device())
        return obj2dict(devprops)
    except:
        return None

paramtest = [
    [4,4,3,"random"]
    ,[10, 5, 2, 'random']
    ,[14, 1, 3, 'random']
    # ,[3, 3, 0, 'grid2d']
    # ,[8, 4, 0, 'line']
]

def mean_mmax(x: list):
    mx, mn = max(x), min(x)
    x.remove(mx)
    x.remove(mn)
    return np.mean(x)

def format_flops(flops):
    ord = 3*int(np.log10(flops)/3)
    suffix = {
        3: 'k'
        ,6: 'M'
        ,9: 'G'
        , 12: 'T'
    }[ord]
    return f'{(flops/10**ord).round(2)}{suffix}'


'''
Purpose: For every test problem, generate one fixed contraction peos
'''
def get_fixed_peos_for_a_pb(G, gamma, beta, algo:str, sim):
    peos, widths = [], []
    for edge in G.edges:
        cir = sim._edge_energy_circuit(G, gamma, beta, edge)
        tn = qtensor.optimisation.TensorNet.QtreeTensorNet.from_qtree_gates(cir)
        opt = qtensor.toolbox.get_ordering_algo(algo)
        peo, _ = opt.optimize(tn)
        width = opt.treewidth
        peos.append(peo)
        widths.append(width)
    return peos, widths



'''
Function: Simulate one lightcone/edge for one time, we dont look for a lightcone aggregation
CHANGE: Rid aggregation methods
'''
def gen_be_lc_report(G, gamma, beta, edge, peo, backend_name, gen_base = 0):

    if backend_name in gpu_backends:
        curr_backend = get_gpu_perf_backend(backend_name)
    else:
        curr_backend = get_cpu_perf_backend(backend_name)
    curr_sim = QAOAQtreeSimulator(QtreeQAOAComposer,backend=curr_backend)
    circuit = curr_sim._edge_energy_circuit(G, gamma, beta, edge)
    curr_sim.simulate_batch(circuit, peo = peo)
    curr_sim.backend.gen_report(show = False)

    '''
    Generate report table data
    '''
    tensor_dims = []
    for i, x in enumerate(curr_sim.backend._profile_results):
        bucket_signature, _ = curr_sim.backend._profile_results[x]
        bucket_size = []
        for tensor in bucket_signature:
            tensor_2_size = [qtreeVar.size for qtreeVar in tensor]
            tensor_dim = np.prod(tensor_2_size)
            bucket_size.append(tensor_dim)
        tensor_dims.append(sum(bucket_size))
    
    report_record = curr_sim.backend.report_table.records
    gen_time = gen_base
    title_record = curr_sim.backend.report_table._title_row()[1:]

    '''
    Modify and concatenate new report table
    '''
    title_record.append("byte")
    title_record.append("gen_time")
    for i, row in enumerate(report_record):
        row.append(tensor_dims[i])
        row.append(gen_time)

    return title_record, report_record


'''
Function: bucket_index -> [repeat] amount of reports for this particular bucket
'''
def collect_reports_for_a_lc(G, gamma, beta, edge, peo, backend_name, repeat, gen_base):
    bucketindex_2_reports = {}
    '''
    Collect each lc's report for [repeat] times
    '''
    for _ in range(repeat):
        titles, report_table = gen_be_lc_report(G, gamma, beta, edge, peo, backend_name, gen_base)
        for i, report in enumerate(report_table):
            if i not in bucketindex_2_reports:
                bucketindex_2_reports[i] = [report]
            else:
                bucketindex_2_reports[i].append(report)
    
    #print(bucketindex_2_reports)
    return titles, bucketindex_2_reports

'''
Function: run above functions, collect reports, and reduce the data
'''
def reduce_bucket_reports(G, gamma, beta, edge, peo, backend_name, repeat, gen_base):
    bi_2_reduced = {}
    titles, bi_2_reports = collect_reports_for_a_lc(G, gamma, beta, edge, peo, backend_name, repeat, gen_base)
    
    for bi, report in bi_2_reports.items():
        bi_redux = []
        t_report = np.transpose(report)
        for attr in t_report:
            bi_redux.append(mean_mmax(attr.tolist()))
        bi_2_reduced[bi] = bi_redux

    return titles, bi_2_reduced

'''
Function: Takes in reduced data form, and process them into json format, with knowledge regarding lightcone index and bucket index
'''
def process_reduced_data(G, gamma, beta, edge, peo, backend_name, problem, repeat, gen_base, lc_index, opt_algo):
    titles, bi_2_reduced = reduce_bucket_reports(G, gamma, beta, edge, peo, backend_name, repeat, gen_base)
    GPU_PROPS = get_gpu_props_json()
    lc_collection = []
    for bi, report in bi_2_reduced.items():
        bi_json_usable = {}
        bi_json_usable["backend"] = backend_name
        bi_json_usable["device_props"] = dict(name=platform.node(), gpu=GPU_PROPS)
        bi_json_usable["lightcone_index"] = lc_index
        bi_json_usable["bucket_index"] = bi
        bi_json_usable["opt_algo"] = opt_algo
        for i, attr in enumerate(titles):
            if attr == "flop":
                bi_json_usable["ops"] = report[i]
            else:
                bi_json_usable[attr] = report[i]
        bi_json_usable["problem"] = {
                    "n" :problem[0] , 
                    "p" :problem[1] ,
                    "d" :problem[2] ,
                    'type': problem[3]
                    }
        bi_json_usable["experiment_group"] = "Angela_nslb_circuit"
        lc_collection.append(bi_json_usable)
    #print(json.dumps(lc_collection, indent = 4))

    return lc_collection
    
        


if __name__ == '__main__':
    gen_sim = QAOAQtreeSimulator(QtreeQAOAComposer)
    backends = ['einsum', 'torch_cpu', 'opt_einsum', 'tr_einsum', 'opt_einsum', 'torch_gpu', 'cupy', 'tr_cupy', 'tr_cutensor'] #'tr_torch'

    my_algo = 'rgreedy_0.05_30'

    for pb in paramtest:

        '''
        Generate fixed peos for a given problem, thus be used for various backends
        '''
        with timing(callback=lambda x: None) as gen_pb:
            n, p, d, ttype = pb
            G, gamma, beta = get_test_problem(n=n,p=p,d=d, type = ttype)
            peos, widths = get_fixed_peos_for_a_pb(G, gamma, beta, algo = my_algo, sim = gen_sim)

        gen_base = gen_pb.result
        agg_reports = []
        for be in backends:
            all_lightcones_report = []
            for i, pack in enumerate(zip(G.edges, peos)):
                edge, peo = pack
                curr_report = process_reduced_data(G, gamma, beta, edge, peo, be, pb, 3, 114514, i, my_algo)
                for c in curr_report:
                    print(json.dumps(c))
                #all_lightcones_report.append(curr_report)
            agg_reports.append(all_lightcones_report)