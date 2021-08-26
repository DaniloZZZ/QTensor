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
from qtree.optimizer import Var

gpu_backends = ['torch_gpu', 'cupy', 'tr_torch', 'tr_cupy', 'tr_cutensor']
cpu_backends = ['einsum', 'torch_cpu', 'mkl', 'opt_einsum', 'tr_einsum', 'opt_einsum']

timing = pyrofiler.timing


@lru_cache
def get_test_problem(n=10, p=2, d=3, type='random'):
    print('Test problem: n, p, d', n, p, d)
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
    ,[4, 4, 3, 'random']
    ,[10, 5, 2, 'random']
    ,[14, 1, 3, 'random']
    ,[3, 3, 0, 'grid2d']
    ,[8, 4, 0, 'line']
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
Function: Simulate one lightcone/edge for one time, one simulator repetively using again and again
Issue:    Needs to find the correct way of defining the gen_time
'''
def gen_be_lc_report(G, gamma, beta, edge, peo, backend_name, gen_base = 0):
    with timing(callback = lambda x: None) as gen:
        if backend_name in gpu_backends:
            curr_backend = get_gpu_perf_backend(backend_name)
        else:
            curr_backend = get_cpu_perf_backend(backend_name)
        curr_sim = QAOAQtreeSimulator(QtreeQAOAComposer,backend=curr_backend)
        circuit = curr_sim._edge_energy_circuit(G, gamma, beta, edge)
    curr_sim.simulate_batch(circuit, peo = peo)
    curr_sim.backend.gen_report(show = False)

    tensor_dims = []
    for i, x in enumerate(curr_sim.backend._profile_results):
        bucket_signature, _ = curr_sim.backend._profile_results[x]
        for tensor in bucket_signature:
            tensor_2_size = [qtreeVar.size for qtreeVar in tensor]
            tensor_dim = np.prod(tensor_2_size)
            tensor_dims.append(tensor_dim)
    
    report_record = curr_sim.backend.report_table.records
    gen_time = gen.result + gen_base
    mean_record = np.mean(report_record, axis = 0)
    max_record = np.amax(report_record, axis = 0)
    sum_record = np.sum(report_record, axis = 0)
    title_record = curr_sim.backend.report_table._title_row()[1:]
    byte = np.sum(tensor_dims)
    num_buckets = len(curr_backend.report_table.records)

    # Give Raw OPs
    #print(sum(bs_arr))


    # flop_index = title_record.index("flop")
    # print(sum(list(np.transpose(report_record)[flop_index])))
    # print(sum(bs_arr))
    return max_record, mean_record, sum_record, title_record, byte, gen_time, num_buckets


'''
Function: Generate a collection of above report for [repeat] amount of times, only for the [lc_index] lightcone
'''
def collect_reports_for_a_lc(G, gamma, beta, edge, peo, backend_name, repeat, lc_index, gen_base):
    means_frame = []
    sums_frame = []
    maxs_frame = []
    byte_frame = []
    gen_time_frame = []
    bucket_num_frame = []
    title_2_data = {}

    for _ in range(repeat):
        maxs, means, sums, titles, byte, gen_time, num_buckets = gen_be_lc_report(G, gamma, beta, edge, peo, backend_name, gen_base)
        maxs_frame.append(maxs)
        means_frame.append(means)
        sums_frame.append(sums)
        byte_frame.append(byte)
        gen_time_frame.append(gen_time)
        bucket_num_frame.append(num_buckets)

    means_frame = np.array(means_frame)
    means_frame = np.transpose(means_frame)
    sums_frame = np.array(sums_frame)
    sums_frame = np.transpose(sums_frame)
    maxs_frame = np.array(maxs_frame)
    maxs_frame = np.transpose(maxs_frame)
    for i, title in enumerate(titles):
        title_2_data["mean_"+title] = list(means_frame[i])
        title_2_data["sum_"+title] = list(sums_frame[i])
        title_2_data["max_"+title] = list(maxs_frame[i])
    
    title_2_data["bytes"] = byte_frame
    title_2_data["gen_time"] = gen_time_frame
    title_2_data["bucket_num"] = bucket_num_frame
    title_2_data["lightcone_index"] = lc_index
    
    return title_2_data


'''
Function: Create a copy and reduce the above collection of reports for one lightcone by applying mean_mmax
'''
def reduce_reports_for_a_lc(collection:dict):
    collection_c = collection.copy()
    for key, value in collection_c.items():
        if key != "lightcone_index":
            collection_c[key] = mean_mmax(value)
    return collection_c


'''
Function: Collect each lc's reduced report and form an array with a length of [# of lightcones]
NOTE: MAY NOT BE NEEDED, AS WE DO NOT NEED ARRAY
'''
def merge_all_lightcones_report(list_of_dicts:list):
  merged = {}
  for r in list_of_dicts:
    for key, value in r.items():
      if key not in merged:
        merged[key] = [value]
      else:
        merged[key].append(value)
  return merged


'''
Function: Reduce the merged report according to the keyword
NOTE: MAY NOT BE NEEDED
'''
def reduce_merged_report(merged:dict):
    #print(merged)
    merged_reduced = {}
    for key, value in merged.items():
        if key != "lightcone_index":
            if key[:3] == "mea":
                merged_reduced[key] = np.mean(value)
            elif key[:3] == "max":
                merged_reduced[key] = np.amax(value)
            else:
                if key == "sum_time":
                    merged_reduced["sum_time_std"] = np.std(value)
                merged_reduced[key] = np.sum(value)
    return merged_reduced


'''
Function: Generate json report for current backend and problem
TODO: 
'''
def gen_json_for_be_pt(backend_name: str, problem:list, redux_report: dict, opt_algo:str, task_type = "QAOAEnergyExpectation"):
    GPU_PROPS = get_gpu_props_json()
    res = dict(
        backend = backend_name,
        device_props = dict(name=platform.node(), gpu=GPU_PROPS),
        task_type = task_type,
        opt_algo = opt_algo,
        lightcone_index = redux_report["lightcone_index"],
        flops = redux_report["mean_FLOPS"],
        flops_str = format_flops(redux_report["mean_FLOPS"]),
        ops = redux_report["sum_flop"],
        width = redux_report["max_max_size"],
        mult_time = redux_report["sum_time"],
        #mult_relstd = np.std(redux_report["sum_time"]),
        bytes = redux_report["bytes"],
        gen_time = redux_report["gen_time"],
        num_buckets = redux_report["bucket_num"],
        problem = {
                    "n" :problem[0] , 
                    "p" :problem[1] ,
                    "d" :problem[2] ,
                    'type': problem[3]
                    },
        experiment_group = "Angela_nslb_circuit_sl"
        # add a field for lightcone_index
    )
    return res




if __name__ == '__main__':
    gen_sim = QAOAQtreeSimulator(QtreeQAOAComposer)
    backends = ['einsum', 'torch_cpu', 'opt_einsum', 'tr_einsum', 'opt_einsum', 'torch_gpu', 'cupy', 'tr_cupy', 'tr_cutensor'] #'tr_torch'
    my_algo = 'rgreedy_0.05_30'

    for pb in [paramtest[0]]:

        '''
        Generate fixed peos for a given problem, thus be used for various backends
        '''
        with timing(callback=lambda x: None) as gen_pb:
            n, p, d, ttype = pb
            G, gamma, beta = get_test_problem(n=n,p=p,d=d, type = ttype)
            peos, widths = get_fixed_peos_for_a_pb(G, gamma, beta, algo = my_algo, sim = gen_sim)

        gen_base = gen_pb.result
        for be in backends:
            '''
            Collecting all lightcones' reduced report
            '''
            all_lightcones_report = []
            for i, pack in enumerate(zip(G.edges, peos)):
                edge, peo = pack
                curr_lightcone_report = collect_reports_for_a_lc(G, gamma, beta, edge, peo, be, 3,i, gen_base)
                reduced_lightcone_report = reduce_reports_for_a_lc(curr_lightcone_report)
                ''''
                DESIGN CHOICE:
                convert reduced_lightcone_report into a json-usable formart here
                then append to all_lightcone_report
                '''
                js_usable = gen_json_for_be_pt(be, pb, reduced_lightcone_report, my_algo)
                #print(js_usable)
                all_lightcones_report.append(js_usable)
                #print("{}th lc finished REEEEEEEEEEEEEE".format(i))

            
            print(json.dumps(all_lightcones_report, indent=4))

            ''''
            UNCOMMENT THE BELOW FOR FINAL USEAGE
            '''
            #merged_report = merge_all_lightcones_report(all_lightcones_report)

            '''
            Reduced Merged Report, good for writing the final json
            '''
            #merged_report_reduced = reduce_merged_report(merged_report)


            '''
            generate report for current backend and problem
            '''
            #final = json.dumps(gen_json_for_be_pt(be, pb, merged_report, opt_algo= my_algo), indent = 4)
            #print(final)
            