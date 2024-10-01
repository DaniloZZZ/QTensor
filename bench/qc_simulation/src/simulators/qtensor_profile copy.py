import sys
sys.path.append('.')
import time
from test_circuits import gen_qaoa_maxcut_circuit
import qtensor
import qtree
from qtensor.merged_indices.transposed_bucket_elimination import bucket_elimination
import numpy as np
import pandas as pd
import pyrofiler

from qtensor import QtreeQAOAComposer
from qtensor import QtreeSimulator
from qtensor import toolbox
from qtensor.contraction_backends import get_backend, PerfBackend

def bucket_contraction_report(tn, buckets, backend,
                              bucket_elimination=bucket_elimination
                             ):
    """
    Returns:
        str: string summary of report
        report_table: table of metrics for each bucket contraction step
        profile_results: raw bucket data: dict mapping `str: (indices, time)`
    """

    perf_backend = PerfBackend(print=False, num_lines=20)
    perf_backend.backend = backend
    result = bucket_elimination(
        buckets, perf_backend.process_bucket,
        n_var_nosum=len(tn.free_vars)
    )
    result_data = perf_backend.get_result_data(result).flatten()
    print("Result data:", result_data)
    # compute report_table
    rep_txt = perf_backend.gen_report(show=False)
    return perf_backend.report_table

def get_buckets_tn(circ, backend, ordering_algo:str, batch_vars=0, seed=10):
    np.random.seed(seed)
    tn = qtensor.optimisation.QtreeTensorNet.from_qtree_gates(circ)
    opt = toolbox.get_ordering_algo(ordering_algo)
    sim = QtreeSimulator(optimizer=opt, backend=backend)
    sim.prepare_buckets(circ, batch_vars=batch_vars)
    return sim.buckets, tn

'''
Function: Generate a collection of above report, and process them into final usable form
I/O: ... -> processed data is a dict, directly usable by json
'''
def collect_process_be_pt_report(repeat: int, backend, circ, ordering_algo='greedy'):
    timing = pyrofiler.timing
    with timing(callback=lambda x: None) as gen:
        buckets, tn = get_buckets_tn(circ, backend, ordering_algo, batch_vars=0)

    tables = []
    wall_start = time.time()
    for _ in range(repeat):
        b_copy = [l.copy() for l in buckets]
        table = bucket_contraction_report(tn, b_copy, backend)
        tables.append(table)
    wall_end = time.time()
    print("Wall time per run: ", (wall_end - wall_start) / repeat)
    report = pd.concat(tables, keys=range(repeat), names=['repeat', 'step'])
    return report


def mean_mmax(x: list):
    mx, mn = max(x), min(x)
    x.remove(mx)
    x.remove(mn)
    return np.mean(x)

def main():
    Ns = [30]
    p = 10
    ordering_algo = 'greedy'
    repeats = 2
    top_K = 15
    backend_name = 'torch_cpu'
    print("backend: ", backend_name)
    for N in Ns:
        print(f"N={N}")
        backend = get_backend(backend_name)
        circ = gen_qaoa_maxcut_circuit(N, p)
        report = collect_process_be_pt_report(repeats, backend, circ, ordering_algo=ordering_algo)

        stats = report[["time"]].groupby('step').agg(['mean', 'min', 'max', 'std'])
        stats = pd.concat([
            stats,
            report[["flop","FLOPS", 'result_size', 'bucket_len']].groupby('step').agg(['mean']),
        ], axis=1)
        stats.sort_values(by=[('time', 'mean')], ascending=False, inplace=True)
        print(f"Top {top_K} steps by time:")
        print(stats.head(top_K))
        print(f"Top {top_K} bucket info:")

        ixs = report['indices'].groupby('step').first()
        for i in stats.head(top_K).index:
            print(f"Step {i}: {ixs.loc[i]}")
        print("Time by bucket size:")

        stats = pd.concat([
            report[["time"]].groupby('step').agg('mean'),
            report[["flop","FLOPS", 'result_size', 'bucket_len']].groupby('step').first()
        ], axis=1)
        print(stats[['time', 'result_size', 'FLOPS']].groupby('result_size').agg(['mean', 'sum', 'count']))
        print("Total time:")
        print(stats['time'].sum())

if __name__=="__main__":
    main()


