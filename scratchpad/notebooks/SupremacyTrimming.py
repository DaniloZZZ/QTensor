import qtree
import qtensor
import qtensor.optimisation as qopt
from qtensor.optimisation.Optimizer import TamakiTrimSlicing
import numpy as np
import networkx as nx
import networkx.algorithms.approximation as nx_approx
import matplotlib.pyplot as plt
import glob

files = glob.glob('../../qtree/test_circuits/inst/bristlecone/cz_v2/*/*_0.txt')         
file = files[1]

n_qubits, qc = qtree.operators.read_circuit_file(file)
print(f'Qubits: {n_qubits}, gates: {len(sum(qc, []))}')

indices = set(g._qubits for g in sum(qc, []))
#indices
devmap = nx.Graph()
for i in indices:
    if len(i)==1:
        devmap.add_node(i[0])
    elif len(i)==2:
        devmap.add_edge(*i)
        
nx.draw_kamada_kawai(devmap, with_labels=True)

tn = qtensor.optimisation.TensorNet.QtreeTensorNet.from_qtree_gates(sum(qc,[]))
line_graph = tn.get_line_graph()
nx.draw_kamada_kawai(line_graph, node_size=1)

opt = TamakiTrimSlicing()
opt.max_tw = 32

def approx_time(opt):
    return 2**opt.treewidth/1e8 * 2**len(opt.parallel_vars)

peo, par_vars, tn = opt.optimize(tn)
print(f'Approximate simulation time: {approx_time(opt)}s, Treewith: {opt.treewidth}, Paths: {2**len(par_vars)}')