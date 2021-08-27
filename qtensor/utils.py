import copy
import time
import numpy as np
import itertools
import qtree
import matplotlib.pyplot as plt
import networkx as nx
from tqdm.auto import tqdm
import functools
import operator
from collections import defaultdict
from typing import List, Tuple, Collection, TypeVar, Iterable


def bucket_indices(bucket):
    return merge_sets(set(x.indices) for x in bucket)


T = TypeVar('T', set, frozenset)
def merge_sets(sets: Iterable[T]) -> T:
    """ Merges an iterable of sets. """
    first, *other = sets
    return first.union(*other)

# -- Merged indices

T = TypeVar('T', bound=Collection)
def largest_merges(vsets: List[T], merged_ix: List) -> Tuple[List[T], set]:
    """ Finds a largest set of indices such that index is 
    1. specific to exactly 2 tensors
    2. part of merged_ix
    """
    vsets = sorted(vsets, key=len, reverse=True) # from large to small
    vsets = [vs for vs in vsets if len(vs)>1]
    v_to_t = defaultdict(list)
    for t in vsets:
        for v in t:
            if v in merged_ix:
                v_to_t[v].append(t)
    #print('vtot', v_to_t)

    tt_to_vv = defaultdict(list)
    for v, tt in v_to_t.items():
        tt_to_vv[tuple(tt)].append(v)

    #print('tt to vv', tt_to_vv)

    have_two = [tt for tt in tt_to_vv if len(tt)<3]
    verts_2t = [tt_to_vv[tt] for tt in have_two]
    merged_ix_set = set(merged_ix)
    verts_2t_contr = [set(x).intersection(merged_ix_set) for x in verts_2t]
    #print('vsets', vsets, merged_ix)
    #print('have two or 1', verts_2t, have_two)
    if len(verts_2t)==0:
        return [set(), set()], set()
    sorted_mergeable = sorted(verts_2t_contr, key=len, reverse=True)
    selected = sorted_mergeable[0]
    # just take the tensors that correspond to arbitrary index in selected
    tensors_contr = v_to_t[next(iter(selected))]
    return tensors_contr, selected


def find_mergeable_indices(peo, buckets):
    """
    Buckets should be ordered
    Args:
        peo: list of vertices
        vsets: list of lists of lists of vertices
    Returns:
        merged_ix: list of lists
        width: list of sizes of tensor per contraction
    """
    contraction_widths = []
    to_peo_inds = lambda x: peo.index(x)
    vsets = [[frozenset(map(to_peo_inds, t)) for t in bucket] for bucket in buckets] + [[frozenset()]]
    merged_ix = []
    i = 0
    while i < len(peo):
        merged_ix.append([i])
        merged_bucket = vsets[i]
        next_vset = merge_sets(vsets[i])
        #print(next_vset)
        if i<len(peo)-1:
            while all(vs.issubset(next_vset) for vs in vsets[i+1]):
                merged_ix[-1].append(i+1)
                merged_bucket += vsets[i+1]
                i += 1
                #next_vset = merge_sets([next_vset] + list(vsets[i]))
                #print('m', peo[i], next_vset)
                contraction_widths.append(0)
                if i == len(peo)-1:
                    break
        i += 1

        if len(merged_ix[-1])>1:
            _, contracted = largest_merges(merged_bucket, merged_ix[-1])
            contraction_widths.append(len(next_vset) - len(contracted))
        else:
            contraction_widths.append(len(next_vset))
        next_vset -= set(merged_ix[-1])
        #contraction_widths.append(len(next_vset))
        if len(next_vset):
            min_ix = min(next_vset)
            vsets[min_ix].append(next_vset)
            #print('append', next_vset)

    return merged_ix, contraction_widths

# -- 


def vertex_is_simplical(graph, vertex):
    # Get neighbors with accounting for self-loops
    neighbors = set(graph.neighbors(vertex)) - set((vertex, ))
    clique = itertools.combinations(neighbors, 2)
    edges = set(graph.edges(nbunch=neighbors))
    return all(edge in edges for edge in clique)

def contraction_steps(old_graph,  peo=None):
    """ Eliminate all verticies in graph that have degree
    smaller than `min_degree`
    Works in-place
    """
    graph = old_graph.copy()
    if peo is None:
        peo = sorted(graph.nodes(), key=int)

    steps = []
    joinstr = lambda x: ''.join(str(y) for y in x)
    if not isinstance(peo[0], str):
        raise Exception('only chars are supported for now')
    for node in peo:
        neighbors = joinstr(set(graph.neighbors(node))-set([node]))
        ixs = set([joinstr(tensor['indices'])
                   for *_, tensor
                   in graph.edges(node, data='tensor')])
        steps.append((
            node
           ,len(neighbors)
           ,vertex_is_simplical(graph, node)
           ,ixs
           ,neighbors))
        qtree.graph_model.eliminate_node(graph, node)
    return steps


def eliminate_low_degrees(graph, min_degree=3):
    """ Eliminate all verticies in graph that have degree
    smaller than `min_degree`
    Works in-place
    """
    to_eliminate = []

    for node, degree in graph.degree:
        if degree < min_degree:
            to_eliminate.append(node)
    for node in to_eliminate:
        eliminate_node_no_structure(graph, node)
    return len(to_eliminate)


def get_neighbors_peo_vars(old_graph, inplace=False):
    if inplace:
        graph = old_graph
    else:
        graph = copy.deepcopy(old_graph)
    graph.remove_edges_from(list(nx.selfloop_edges(old_graph)))
    peo = []
    nghs = []

    while graph.number_of_nodes():
        ###start = time.time()
        costs = np.array(list(
            map(len, map(operator.itemgetter(1), graph.adjacency()))
        ))
        #costs = list(graph.degree)
        ###costtime = time.time() - start

        ###start = time.time()
        best_idx = np.argmin(costs)
        best_degree = costs[best_idx]
        best_node = list(graph.nodes())[best_idx]
        del costs
        peo.append(best_node)
        nghs.append(best_degree)
        #nodeiter_time = time.time() - start


        #start = time.time()
        #qtree.graph_model.eliminate_node(graph, best_node)
        eliminate_node_no_structure(graph, best_node)
        #eltime = time.time() - start
        #pbar.set_postfix(costiter=1/costtime, nodeiter=1/nodeiter_time, eliter=1/eltime ,costtime=costtime, nodeiter_time=nodeiter_time, eltime=eltime)
    return peo, nghs

def get_neighbors_peo(old_graph):
    graph = copy.deepcopy(old_graph)
    graph.remove_edges_from(nx.selfloop_edges(old_graph))
    peo = []
    nghs = []

    while graph.number_of_nodes():
        ###start = time.time()
        costs = np.array(list(
            map(len, map(operator.itemgetter(1), graph.adjacency()))
        ))
        #costs = list(graph.degree)
        ###costtime = time.time() - start

        ###start = time.time()
        best_idx = np.argmin(costs)
        best_degree = costs[best_idx]
        best_node = list(graph.nodes())[best_idx]
        del costs
        peo.append(int(best_node))
        nghs.append(int(best_degree))
        #nodeiter_time = time.time() - start


        #start = time.time()
        #qtree.graph_model.eliminate_node(graph, best_node)
        eliminate_node_no_structure(graph, best_node)
        #eltime = time.time() - start
        #pbar.set_postfix(costiter=1/costtime, nodeiter=1/nodeiter_time, eliter=1/eltime ,costtime=costtime, nodeiter_time=nodeiter_time, eltime=eltime)
    return peo, nghs

def eliminate_node_no_structure(graph, node):
    neighbors_wo_node = set(graph[node])
    while node in neighbors_wo_node:
        neighbors_wo_node.remove(node)

    graph.remove_node(node)

    # prepare new tensor
    if len(neighbors_wo_node) > 1:
        graph.add_edges_from( itertools.combinations(neighbors_wo_node, 2))


def get_locale_peo(old_graph, rule):
    # This is far below computationally effective
    graph = copy.deepcopy(old_graph)
    
    path= []
    vals = []

    with tqdm(total=graph.number_of_nodes(), desc='Node removal') as pbar:
        while graph.number_of_nodes():
            #nodes = sorted(graph.nodes, key=int)
            nodes = sorted(list(graph.nodes), key=int)
            rule_ = lambda n: rule(graph, n)
            start = time.time()
            costs = list(map(rule_, nodes))
            costtime = time.time() - start
            _idx = np.argmin(costs)
            vals.append(costs[_idx])
            node = nodes[_idx]
            path.append(node)
            start = time.time()
            eliminate_node_no_structure(graph, node)
            eltime = time.time() - start
            pbar.update(1)
            pbar.set_postfix(eliter=1/eltime, costiter=1/costtime, degree=costs[_idx])
    return path, vals


def get_test_circ_filename(root, size, depth=10, id_=0):
    grid = f'{size}x{size}'
    return f'{root}/inst/cz_v2/{grid}/inst_{grid}_{depth}_{id_}.txt'

def reorder_graph(graph, peo):
    graph, label_dict = qtree.graph_model.relabel_graph_nodes(
        graph, dict(zip(peo, sorted(graph.nodes(), key=int)))
    )
    return graph, label_dict


def plot_cost(mems, flops):
    plt.yscale('log')
    ax = plt.gca()
    ax.grid(which='minor', alpha=0.5, linestyle='-', axis='both')
    ax.grid(which='major', alpha=0.6, axis='both')
    ax.yaxis.set_tick_params(labelbottom=True)
    #ax.minorticks_on()

    plt.plot(mems, label='Memory')
    plt.plot(flops, label='FLOP')
    plt.legend()


def nodes_to_vars(old_graph, peo):
    peo_vars = [qtree.optimizer.Var(v, size=old_graph.nodes[v]['size'],
                    name=old_graph.nodes[v]['name']) for v in peo]
    return peo_vars


def n_neighbors(graph, node):
    return len(list(graph[node].values()))

def degree(graph, node):
    return graph.degree(node)


def edges_to_clique(graph, node):
    N = graph.degree(node)
    edges = graph.edges(node)
    return N*(N-1)//2 - len(edges)


def _neighbors(graph, node):
    return list(graph.neighbors(node))


def get_neighbors_path(old_graph, peo=None):
    if peo is not None:
        graph, _ = reorder_graph(old_graph, peo)
    else:
        graph = copy.deepcopy(old_graph)
    ngh = []
    nodes = sorted(graph.nodes, key=int)
    for node in nodes:
        ngh.append(n_neighbors(graph, node))
        #qtree.graph_model.eliminate_node(graph, node)
        eliminate_node_no_structure(graph, node)
    return nodes, ngh

def nodes_at_distance(G, nodes, dist):
    nodes = list(nodes)
    for d in range(dist):
        range_d_nodes = []
        for n in nodes:
            neigh = list(G[n].keys())
            range_d_nodes += neigh
        nodes += range_d_nodes
        nodes  = list(set(nodes))
    return set(nodes)

def get_edge_subgraph_old(G, edge, dist):
    nodes = nodes_at_distance(G, edge, dist)
    return G.subgraph(set(nodes))

def nodes_group_by_distance(G, nodes, dist):
    nodes_groups = { 0: list(nodes) }
    inner_circle = list(nodes)

    for d in range(1, dist+1):
        range_d_nodes = []
        for n in nodes_groups[d-1]:
            neigh = list(G.neighbors(n))
            range_d_nodes += neigh
        range_d_set = set(range_d_nodes)
        nodes_groups[d] = list(range_d_set - set(inner_circle))
        inner_circle += nodes_groups[d]
    return nodes_groups


def get_edge_subgraph(G, edge, dist):
    nodes_groups = nodes_group_by_distance(G, edge, dist)
    all_nodes = sum(nodes_groups.values(), [])
    subgraph = G.subgraph(all_nodes).copy()
    farthest_nodes = nodes_groups[dist]
    #   for v in farthest_nodes:
    #       u, w = edge
    #       shpu, shpw = nx.shortest_path(G, u, v), nx.shortest_path(G, w, v)
    #       print('shp, dist', len(shpu), len(shpw), dist)
    #       assert (len(shpu) == dist + 1) or (len(shpw) == dist+1)
    edges_to_delete = []
    for u, v in subgraph.edges():
        if (u in farthest_nodes) and (v in farthest_nodes):
            edges_to_delete.append((u,v))
    #print('removing edges', edges_to_delete)
    subgraph.remove_edges_from(edges_to_delete)
    return subgraph


class ReportTable():
    measures = {
        'max':np.max
        ,'min': np.min
        ,'mean': np.mean
        ,'sum': np.sum
        ,'median': np.median
    }
    def __init__(self, measure=['max', 'min'], columns=[], max_records=100):
        self.measure = measure
        self.columns = columns
        self.records = []
        self.max_records = max_records

    def record(self, **kwargs):
        if self.columns:
            if set(self.columns) != set(kwargs.keys()):
                raise ValueError(f"columns doesn't match: {kwargs.keys()}, expect: {self.columns}")
        else:
            self.columns = set(kwargs.keys())
        self.records += [[kwargs[key] for key in self.columns]]

    def _title_row(self):
        return ['N'] + list(self.columns)

    def _print_title(self):
        print(','.join(self._title_row()))

    def _format_row(self, row):
        def format(x):
            if isinstance(x, str): return x
            if x == 0:
                return '0'
            if x > 1000 or x<0.001:
                return f"{x:.3e}"
            if isinstance(x, int):
                return str(x)
            return f"{x:.3f}"
        return [format(x) for x in row]

    def _measure_rows(self):
        rows = []
        for stat_label in self.measure:
            stat_func = self.measures[stat_label]
            stat_row = [stat_label]
            for i, key in enumerate(self.columns):
                column = [row[i] for row in self.records]
                stat_row.append(stat_func(column))
            rows.append(stat_row)
        return rows

    def _print_row(self, row):
        print(','.join(self._format_row(row)))

    def markdown(self):
        cells = []
        cells.append(self._title_row())
        for i, row in enumerate(self.records):
            cells.append([str(i)] + self._format_row(row))
        cells += [self._format_row(r) for r in self._measure_rows()]
        table = MarkdownTable(cells)
        return table.markdown()

    def print(self):
        self._print_title()
        for i in range(min(self.max_records, len(self.records))):
            self._print_row([i] + list(self.records[i]))

        for stat_row in self._measure_rows():
            self._print_row(stat_row)

class MarkdownTable:
    """ Stolen from https://github.com/lzakharov/csv2md/blob/master/csv2md/table.py """
    def __init__(self, cells):
        self.cells = cells
        self.widths = list(map(max, zip(*[list(map(len, row)) for row in cells])))

    def markdown(self, center_aligned_columns=None, right_aligned_columns=None):
        def format_row(row):
            return '| ' + ' | '.join(row) + ' |'

        rows = [format_row([cell.ljust(width) for cell, width in zip(row, self.widths)]) for row in self.cells]
        separators = ['-' * width for width in self.widths]

        if right_aligned_columns is not None:
            for column in right_aligned_columns:
                separators[column] = ('-' * (self.widths[column] - 1)) + ':'
        if center_aligned_columns is not None:
            for column in center_aligned_columns:
                separators[column] = ':' + ('-' * (self.widths[column] - 2)) + ':'

        rows.insert(1, format_row(separators))

        return '\n'.join(rows)

    @staticmethod
    def parse_csv(file, delimiter=',', quotechar='"'):
        return Table(list(csv.reader(file, delimiter=delimiter, quotechar=quotechar)))

