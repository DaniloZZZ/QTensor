import qtensor
from qtensor.optimisation.kahypar_ordering import generate_TN
from qtensor.optimisation.kahypar_ordering import use_kahypar

def test_dual_hg():
    hg = {1: [1, 2, 3], 2: [1, 4], 3: []}
    dual_expect = {1: [1, 2], 2: [1], 3: [1], 4: [2]}
    dual = generate_TN.dual_hg(hg)
    assert dual==dual_expect

def test_tn():
    import networkx as nx
    N = 5
    g = nx.path_graph(N)
    """
    Resulting TN (hypergraph):
        --- input ---

        |    |    |
        M    M    M
        |    |    |
        H    H    H
        |\  /|\  /|
         -ZZ- -ZZ- -
        |/  \|/  \|
        U    U    U
        |    |    |
        M    M    M
        |    |    |

        --- output ---

    """
    dangling_cnt = 2*N
    vert_cnt = 4*N + (N-1)
    edge_cnt = 5*N

    comp = qtensor.DefaultQAOAComposer(g, gamma=[1], beta=[2])
    comp.ansatz_state()
    circ = comp.circuit
    tn_ = qtensor.optimisation.QtreeTensorNet.from_qtree_gates(circ)
    print('tensors', tn_.tensors)

    tn = generate_TN.circ2tn(circ)
    #dual = generate_TN.dual_hg(tn)
    print(tn)
    dangling = [item for item in tn.values() if len(item)==1]

    assert len(dangling) == dangling_cnt
    verts = set(sum(tn.values(), []))
    assert len(verts) == vert_cnt
    assert len(tn) == edge_cnt

def test_Kahypar(circ, compare = True, plot = False):
    ### test treewidth
    from qtensor.optimisation.Optimizer import TamakiOptimizer, GreedyOptimizer, KahyparOptimizer
    from qtensor import utils
    old_tn=qtensor.optimisation.QtreeTensorNet.from_qtree_gates(circ)
    line_graph = old_tn.get_line_graph()   
    
    ### use Kahypar
    kahypar_opt = KahyparOptimizer()
    with timing() as t_kahypar:
        kahypar_peo, _ = kahypar_opt.optimize(circ)
    kahypar_contraction_width = kahypar_opt.treewidth
    print(kahypar_peo)
    
    if compare == True: 
        ### compare with random 
        # from itertools import permutations
        # from random import choice
        # permutation_list = list(permutations(range(min(peo), max(peo)+1))
        # sequence = [i for i in range(len(permutation_list))]
        contraction_width_list = []; 
        import numpy as np
        np.random.seed(1)
        with timing() as t_random:
            for _ in range(1):
                random_peo=np.random.permutation(kahypar_peo)
                nodes, ngh = utils.get_neighbors_path(line_graph, list(random_peo))
                contraction_width_list.append(max(ngh))
                
        
        ### compare with tamaki 
        tamaki_opt = TamakiOptimizer(wait_time=1) # time to run tamaki, in seconds
        with timing() as t_tamaki:
            tamaki_peo, _ = tamaki_opt.optimize(old_tn)
        #tamaki_peo = tamaki_peo[2*N:len(tamaki_peo)]
        #nodes, ngh = utils.get_neighbors_path(line_graph, tamaki_peo)
        tamaki_contraction_width = tamaki_opt.treewidth
        
        
        ### compare with greedy 
        greed_opt = GreedyOptimizer()
        greedy_peo, _ = greed_opt.optimize(old_tn)
        #greedy_peo = greedy_peo[2*N:len(greedy_peo)]
        with timing() as t_greedy:
            greedy_peo, _ = greed_opt.optimize(old_tn)
        #nodes, ngh = utils.get_neighbors_path(line_graph, greedy_peo)
        greedy_contraction_width = greed_opt.treewidth
        print('--Width--')
        print(f'Partition contraction width: {kahypar_contraction_width}') 
        print(f'Min random contraction width: {min(contraction_width_list)}') 
        print(f'Tamaki contraction width: {tamaki_contraction_width}')
        print(f'Greedy contraction width: {greedy_contraction_width}')
        print()
        
        print('--Time--')
        print(f'Kahypa: {t_kahypar.result}') 
        print(f'Random: {t_random.result}') 
        print(f'Tamaki: {t_tamaki.result}')
        print(f'Greedy: {t_greedy.result}')
        print()
        
        if plot == True:   
            import matplotlib.pyplot as plt  
            import qtree  
            # for line graph
            def plot_costs_peo(graph, peo, print_stat=False):
                graph, _ = qtensor.utils.reorder_graph(graph, peo)
                mems, flops = qtree.graph_model.get_contraction_costs(graph)
                qtensor.utils.plot_cost(mems, flops)
                if print_stat:
                    print(f'Total FLOP: {sum(flops):e}, maximum mem: {max(mems):e}')
                return mems, flops
                    
            def plot_conraction_costs(graph, peo, plot_title=''):
                f, axs = plt.subplots(1, 2, figsize=(10, 4))
                plt.sca(axs[0])
                mems, flops = plot_costs_peo(graph, peo, print_stat=True)
                plt.sca(axs[1])
                _, path = qtensor.utils.get_neighbors_path(graph, peo)
                plt.title(plot_title)
                plt.plot(path)
                plt.grid()
                print(f'Max peo: {max(path)}')
            
            # real mems & flops    
            #mems, flops = qtree.graph_model.get_contraction_costs(line_graph)
            #qtensor.utils.plot_cost(mems,flops)
            plot_conraction_costs(line_graph, greedy_peo)
            plt.title('Greedy cost')
            plot_conraction_costs(line_graph, kahypar_peo)
            plt.title('Cost for partition peo')
            plt.show() # shows a plot if run from terminal on machine with GUI on

    
################--------------------------################
#test_dual_hg()
#test_tn()

################--------------------------################
# -- Timing
from contextlib import contextmanager
import time
@contextmanager
def timing():
    class Ret:
        result=None
    r = Ret()
    start = time.time()
    yield r
    end = time.time()
    r.result = end-start    
# --


### generate dual graph for partition 
#tn ={'v_1': ['A','C'], 'v_2':['A','B'], 'v_3':['B','C','D'], 
#         'v_4':['C','E'], 'v_5':['D','F'], 'v_6':['E','F']}
import networkx as nx
N = 50 # the larger the harder
p = 3 # the larger the harder
#g = nx.path_graph(N) # simple graph structure
g = nx.random_regular_graph(3, N) # more complicated structure
comp = qtensor.DefaultQAOAComposer(g, gamma=[1]*p, beta=[2]*p)
comp.ansatz_state()
circ = comp.circuit
test_Kahypar(circ)

### test in script
# tn = generate_TN.circ2tn(circ)
# # preprocessing to remove edges i_ and o_ (which have only one vertex)
# edge =list(tn.keys()); edge.sort()
# rem_num_list = [*range(N), *range(len(edge)-1, len(edge)-N-1, -1)]
# rem_list = [edge[i] for i in rem_num_list]
# [tn.pop(key) for key in rem_list]
# kwargs = {'K': 5, 'eps': 0.1, 'seed': 2021, 'mode':0, 'objective':0} 
# with timing() as t_kahypar:
#     tn_partite_list = use_kahypar.recur_partition(tn,**kwargs)        
#     order, _ = use_kahypar.tree2order(tn,tn_partite_list) # top to bottom
# #full_order=rem_list; full_order.extend(order)
# print(order)  
# peo = [int(x) for x in order]
# #nodes, ngh = utils.get_neighbors_path(line_graph, peo)
# #contraction_width = max(ngh)

