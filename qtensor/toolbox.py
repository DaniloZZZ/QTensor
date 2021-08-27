import networkx as nx
import numpy as np
from itertools import repeat
from tqdm.auto import tqdm
import time
from multiprocessing.dummy import Pool

from qtensor.optimisation.TensorNet import QtreeTensorNet
from qtensor.optimisation.Optimizer import GreedyOptimizer, TamakiOptimizer, WithoutOptimizer, TamakiTrimSlicing, DefaultOptimizer, KahyparOptimizer

from qtensor.optimisation import RGreedyOptimizer, LateParOptimizer
from qtensor.utils import get_edge_subgraph
from qtensor import QtreeQAOAComposer, OldQtreeQAOAComposer, ZZQtreeQAOAComposer, DefaultQAOAComposer
from qtensor import tools
import qtensor

def bethe_graph(p, degree):
    def add_two_nodes_to_leafs(graph):
        """ Works in-place """
        leaves = [n for n in graph.nodes() if graph.degree(n) <= degree-2]
        n = graph.number_of_nodes()
        for leaf in leaves:
            next_edges = [(leaf, n+x) for x in range(1, degree)]
            graph.add_edges_from(next_edges)
            n += 2
    graph = nx.Graph()
    graph.add_edges_from([(0,1)])
    for i in range(p):
        add_two_nodes_to_leafs(graph)
    return graph

def random_graph(nodes, type='random', **kwargs):
    """
    Generate a random graph
    Parameters:
        nodes: int
            Number of nodes in the graph
        type: enum['random', 'erdos_renyi']
            algorithm to use
        **kwargs:
            keyword arguments to specific algorithm
            usually of form
                seed: int
                degree: int
    """
    if type == 'random':
        return nx.random_regular_graph(n=nodes, d=kwargs['degree']
                                 , seed=kwargs.get('seed'))
    if type == 'erdos_renyi':
        prob_of_edge_add = kwargs['degree']/(nodes-1)
        return nx.erdos_renyi_graph(n=nodes, p=prob_of_edge_add
                                    , seed=kwargs.get('seed'))
    else:
        raise ValueError('Unsupported graph type')

def get_slicing_algo(slicing_algo, par_vars, ordering_algo='default'):
    if 'late-slice' in slicing_algo:
        if '_' in slicing_algo:
            _, bunch_size = slicing_algo.split('_')
            bunches = int(bunch_size)
        else:
            bunches = 1
        optimizer = LateParOptimizer(
            n_bunches=bunches, par_vars=par_vars, ordering_algo=ordering_algo
        )
    else:
        raise ValueError(f'Slicing algorithm not supported: {slicing_algo}')
    return optimizer


def get_ordering_algo(ordering_algo, par_vars=0, **kwargs) -> qtensor.optimisation.Optimizer:
    """ Get optimizer instance from its string specifier. """
    if 'tamaki' in ordering_algo:
        wait_time = 10
        if '_' in ordering_algo:
            params = ordering_algo.split('_')
            wait_time = float(params[-1])
        if 'slice' in ordering_algo:
            max_tw = 25
            optimizer = TamakiTrimSlicing(max_tw=max_tw, wait_time=wait_time, **kwargs)
        else:
            optimizer = TamakiOptimizer(wait_time=wait_time, **kwargs)
    elif 'rgreedy' in ordering_algo:
        if '_' in ordering_algo:
            params = ordering_algo.split('_')
            if len(params) == 2:
                _, temp = ordering_algo.split('_')
                repeats = 10
            else:
                _, temp, repeats = ordering_algo.split('_')
            repeats = int(repeats)
            temp = float(temp)
        else:
            temp = 2
            repeats = 10
        repeats = kwargs.pop('repeats', repeats)
        optimizer = RGreedyOptimizer(temp=temp, repeats=repeats, **kwargs)
    elif ordering_algo == 'greedy':
        optimizer = GreedyOptimizer()
    elif ordering_algo == 'default':
        optimizer = DefaultOptimizer()
    elif ordering_algo == 'naive':
        optimizer = WithoutOptimizer()
    elif ordering_algo == 'kahypar':
        optimizer = KahyparOptimizer()
    else:
        raise ValueError('Ordering algorithm not supported')
    return optimizer

def get_cost_params(circ, ordering_algo='greedy'):

    tn = QtreeTensorNet.from_qtree_gates(circ)
    opt = get_ordering_algo(ordering_algo)

    peo, _ = opt.optimize(tn)
    treewidth = opt.treewidth
    mems, flops = tn.simulation_cost(peo)
    return treewidth, max(mems), sum(flops)



def optimize_circuit(circ, algo='greedy', tamaki_time=15):

    # Should I somomehow generalize the tamaki-time argument? provide something like
    # Optimizer-params argument? How would cli parse this?
    opt = get_ordering_algo(algo)

    tn = QtreeTensorNet.from_qtree_gates(circ)
    peo, tn = opt.optimize(tn)
    return peo, tn, opt

def get_tw(circ, ordering_algo='greedy', tamaki_time=15):
    peo, tn, opt = optimize_circuit(circ, algo=ordering_algo, tamaki_time=tamaki_time)
    treewidth = opt.treewidth
    return treewidth

def get_cost_params(circ, ordering_algo='greedy', overflow_tw=None):
    peo, tn, opt = optimize_circuit(circ, algo=ordering_algo)
    treewidth = opt.treewidth
    print('tw', treewidth)
    if overflow_tw is not None:
        if treewidth > overflow_tw:
            mems, flops = [np.inf], [np.inf]
            return treewidth, np.inf, np.inf
    mems, flops = tn.simulation_cost(peo)
    return treewidth, max(mems), sum(flops)


def qaoa_energy_lightcone_iterator(G, p, max_time=None, composer_type='default'):
    if max_time:
        start = time.time()
    else:
        start = np.inf

    for edge in G.edges():
        circ = qaoa_energy_lightcone_circ(G, p, edge, composer_type=composer_type)
        subgraph = get_edge_subgraph(G, edge, p)
        yield circ, subgraph
        if time.time() - start > max_time:
            break

def qaoa_energy_lightcone_circ(G, p, edge, composer_type='default'):
    gamma, beta = [0.1]*p, [0.3]*p
    if composer_type=='default':
        Composer = DefaultQAOAComposer
    elif composer_type=='cylinder':
        Composer = OldQtreeQAOAComposer
    elif composer_type=='cone':
        Composer = QtreeQAOAComposer
    elif composer_type=='ZZ':
        Composer = ZZQtreeQAOAComposer
    else:
        allowed_composers = ['default', 'cylinder', 'cone', 'ZZ']
        raise Exception(f"Composer type not recognized, use one of: {allowed_composers}")

    composer = Composer(G, beta=beta, gamma=gamma)
    composer.energy_expectation_lightcone(edge)
    return composer.circuit


def qaoa_energy_cost_params_stats_from_graph(G, p, max_time=0, max_tw=None,
                              ordering_algo='greedy', print_stats=False):
    cost_params = []
    tw = mem = flop = 0
    with tqdm(total=G.number_of_edges(), desc='Edge iteration') as pbar:
        for circ, subgraph in qaoa_energy_lightcone_iterator(G, p, max_time=max_time):
            _tw, _m, _f = cost_params(circ, ordering_algo=ordering_algo, overflow_tw=max_tw)
            tw = max(tw, _tw)
            mem = max(mem, _m)
            flop += _f
            pbar.set_postfix(current_tw=tw, subgraph_nodes=subgraph.number_of_nodes())
    return tw, mem, flop


def _twidth_parallel_unit(args):
    circ_graph, ordering_algo, tamaki_time, max_tw = args
    circuit, subgraph = circ_graph
    tw = get_tw(circuit, ordering_algo=ordering_algo, tamaki_time=tamaki_time)
    if max_tw:
        if tw>max_tw:
            print(f'Encountered treewidth of {tw}, which is larger {max_tw}')
            raise ValueError(f'Encountered treewidth of {tw}, which is larger {max_tw}')
    return tw

def _mpi_parallel_unit(args):
    G, p, edge, composer_type, ordering_algo, tamaki_time, max_tw = args
    #graph_arguments, p, edge_index, composer_type, ordering_algo, tamaki_time, max_tw = args
    #G = random_graph(**graph_arguments)
    #edge = list(G.edges)[edge_index]
    start = time.time()
    circuit = qaoa_energy_lightcone_circ(G, p, edge, composer_type=composer_type)
    p1 = time.time()
    tw = get_tw(circuit, ordering_algo=ordering_algo, tamaki_time=tamaki_time)
    p2 = time.time()
    #print(f'Time for circuit creation: {p1-start}, time to find order: {p2-p1}', flush=True)
    return tw

def qaoa_energy_tw_from_graph_mpi(G, p, max_time=0, max_tw=0,
                              ordering_algo='greedy', print_stats=False,
                              tamaki_time=15, composer_type='default'):

    #print('before gen', flush=True)
    #lightcone_gen = qaoa_energy_lightcone_iterator(G, p, max_time=max_time, composer_type=composer_type)
    arggen = zip(G.edges(), repeat(ordering_algo), repeat(tamaki_time), repeat(max_tw))
    arggen = zip(repeat(G), repeat(p), G.edges(), repeat(composer_type),
    #edge indices = range(100_000_000) # should be enough for any reasonable calculation
    #arggen = zip(repeat(graph_arguments), repeat(p), edge_indices, repeat(composer_type),
                repeat(ordering_algo), repeat(tamaki_time), repeat(max_tw))
    twidths = tools.mpi.mpi_map(_mpi_parallel_unit, list(arggen), pbar=True, total=G.number_of_edges())
    if twidths:
        if print_stats:
            tools.mpi.print_stats()
            print(f'med={np.median(twidths)} mean={round(np.mean(twidths), 2)} max={np.max(twidths)}')
        return twidths


def qaoa_energy_tw_from_graph(G, p, max_time=0, max_tw=0,
                              ordering_algo='greedy', print_stats=False,
                              tamaki_time=15, n_processes=1, composer_type='default'):

    lightcone_gen = qaoa_energy_lightcone_iterator(G, p, max_time=max_time, composer_type=composer_type)
    arggen = zip(lightcone_gen, repeat(ordering_algo), repeat(tamaki_time), repeat(max_tw))
    if n_processes > 1:
        print('n_processes', n_processes)
        with Pool(n_processes) as p:
            twidths = list(tqdm(p.imap(_twidth_parallel_unit, arggen), total=G.number_of_edges()))
    else:
        twidths = []
        with tqdm(total=G.number_of_edges(), desc='Edge iteration') as pbar:
            for args in arggen:
                circ_graph, ordering_algo, tamaki_time, max_tw = args
                circuit, subgraph = circ_graph
                tw = _twidth_parallel_unit(args)
                pbar.update()
                pbar.set_postfix(current_tw=tw, subgraph_nodes=subgraph.number_of_nodes())
                twidths.append(tw)

    if print_stats:
        print(f'med={np.median(twidths)} mean={round(np.mean(twidths), 2)} max={np.max(twidths)}')
    return twidths



def qaoa_energy_cost_params_from_graph(G, p, max_time=0, max_tw=0,
                              ordering_algo='greedy', print_stats=False):
    costs = []
    with tqdm(total=G.number_of_edges(), desc='Edge iteration') as pbar:
        for circuit, subgraph in qaoa_energy_lightcone_iterator(G, p, max_time=max_time):
            c = get_cost_params(circuit, ordering_algo=ordering_algo)
            if max_tw:
                if c[0]>max_tw:
                    print(f'Encountered treewidth of {c[0]}, which is larger {max_tw}')
                    break
            costs.append(c)

            pbar.update(1)
            pbar.set_postfix(current_costs=c, subgraph_nodes=subgraph.number_of_nodes())
    if print_stats:
        print(f'med={np.median(costs)} mean={round(np.mean(costs), 2)} max={np.max(costs)}')
    return costs
