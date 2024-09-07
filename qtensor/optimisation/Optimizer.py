import qtensor
import qtree
import psutil
import sys
import numpy as np
import networkx as nx
import copy


from qtensor import utils
from qtensor.optimisation.Greedy import GreedyParvars
from qtensor.optimisation.networkit import greedy_ordering_networkit
from qtensor.optimisation.kahypar_ordering import use_kahypar
from loguru import logger as log


class Optimizer:
    def get_ordering_ints(self, graph, inplace=True):
        raise NotImplementedError

    def _get_ordering(self, graph: nx.Graph, inplace=True):
        """
        Optimize the contraction order for a graph
        Returns:
            peo: list of qtree.Var objects
            path: list of ints
        """
        node_names = nx.get_node_attributes(graph, 'name')
        node_sizes = nx.get_node_attributes(graph, 'size')
        peo, path = self.get_ordering_ints(graph, inplace=inplace)
        # compatibility with slicing
        self.peo_ints = [int(x) for x in peo]

        peo = [qtree.optimizer.Var(var, size=node_sizes[var],
                        name=node_names[var])
                    for var in peo]
        #print('tw=', max(path))
        return peo, path

    def optimize(self, tensor_net):
        """
        Optimize the tensor network.

        Convert tensor network to a graph and find the optimal
        contraction order for bucket elimination.
        """
        graph = tensor_net.get_line_graph()
        free_vars = tensor_net.free_vars
        ignored_vars = tensor_net.ket_vars + tensor_net.bra_vars

        if free_vars:
            # It's more efficient to find ordering in-place to avoid copying
            # We'll need the copy of a graph only if we have free_vars
            print('free vars', free_vars)
            self.free_indices = free_vars
            graph = qtree.graph_model.make_clique_on(graph, free_vars)
            graph_copy = copy.deepcopy(graph)
            self.graph = graph_copy
        else:
            self.free_indices = None

        peo, path = self._get_ordering(graph, inplace=True)
        self.treewidth = max(path)

        if free_vars:
            free_vars_trunk = [v for v in free_vars if int(v) in self.graph.nodes]
            if len(free_vars_trunk) != len(free_vars):
                raise ValueError(f'Free vars were sliced: {free_vars} -> {free_vars_trunk}')
                free_vars = free_vars_trunk
            peo = qtree.graph_model.get_equivalent_peo(self.graph, peo, free_vars)

        peo = ignored_vars + peo
        self.peo = peo
        self.ignored_vars = ignored_vars
        return peo, tensor_net


class WithoutOptimizer(Optimizer):

    def get_ordering_ints(self, graph, inplace=True):
        peo = sorted([int(v) for v in graph.nodes()])
        # magic line
        peo = list(reversed(peo))
        _, path = utils.get_neighbors_path(graph, peo)
        return peo, path

class GreedyOptimizer(Optimizer):
    def get_ordering_ints(self, graph, free_vars=[]):
        #mapping = {a:b for a,b in zip(graph.nodes(), reversed(list(graph.nodes())))}
        #graph = nx.relabel_nodes(graph, mapping)
        peo_ints, path = utils.get_neighbors_peo(graph)

        return peo_ints, path

    def _get_ordering(self, graph, inplace=True):
        node_names = nx.get_node_attributes(graph, 'name')
        node_sizes = nx.get_node_attributes(graph, 'size')
        # performing ordering inplace reduces time for ordering by 60%
        #peo, path = utils.get_neighbors_peo_vars(graph, inplace=inplace)

        # this may be ugly, but it is actually pythonic:)
        # solves two problems: possible inconsistencies in api, and missing networkit.
        # does not introduce overhead

        try:
            peo, path = greedy_ordering_networkit(graph)
        except:
            peo, path = utils.get_neighbors_peo_vars(graph, inplace=inplace)

        peo = [qtree.optimizer.Var(var, size=node_sizes[var],
                        name=node_names[var])
                    for var in peo]
        return peo, path

###################################################################
from qtensor.optimisation.kahypar_ordering import generate_TN
class KahyparOptimizer(Optimizer):
    """
    Properties:
        kahypar_args: dict
    """

    def set_kahypar_kwarge(self, **kwargs):
        """
        Set kahypar parameters

        Example:

        ```
           opt.set_kahypar_kwarge(**{'K': 2, 'eps': 0.1, 'seed': 2021, 'mode':0, 'objective':0}))
        ```
        """
        self.kahypar_args = kwargs

    def get_kahypar_kwarge(self):
        """
        Get kahypar parameters

        Default: {'K': 2, 'eps': 0.1, 'seed': 2021, 'mode':0, 'objective':0}
        """
        if hasattr(self, 'kahypar_args'):
            return self.kahypar_args
        default = {'K': 2, 'eps': 0.1, 'seed': 2021, 'mode':0, 'objective':0}
        return default

    def optimize(self, tensor_net):
        # TODO: move this to get_ordering_ints
        #tensor_net=qtensor.optimisation.QtreeTensorNet.from_qtree_gates(circ)
        #free_vars = tensor_net.free_vars
        ignored_vars = tensor_net.ket_vars + tensor_net.bra_vars
        kwargs = self.get_kahypar_kwarge()
        #tn = generate_TN.circ2tn(circ)
        tn = generate_TN.tn2tn(tensor_net)
        # preprocessing to remove edges i_ and o_ (which have only one vertex)
        #edge =list(tn.keys()); edge.sort()
        #rem_num_list = [*range(N), *range(len(edge)-1, len(edge)-N-1, -1)]
        #rem_list = [edge[i] for i in rem_num_list]
        #[tn.pop(key) for key in rem_list]
        [tn.pop(key) for key in ignored_vars]
        tn_partite_list = use_kahypar.recur_partition(tn,**kwargs)
        peo, _ = use_kahypar.tree2order(tn,tn_partite_list) # top to bottom
        self.peo_ints = [int(x) for x in peo]

        peo = ignored_vars + peo
        line_graph = tensor_net.get_line_graph()
        _, ngh = utils.get_neighbors_path(line_graph, self.peo_ints)

        self.treewidth = max(ngh)
        return peo, tensor_net
###################################################################

class SlicesOptimizer(Optimizer):

    def __init__(self, tw_bias=2, max_tw=None, max_slice=None
                 , base_ordering='greedy'
                 , peo_after_slice_strategy='run-again'
                 , **kwargs):
        self.tw_bias = tw_bias
        self.max_tw = max_tw
        self.max_slice = max_slice
        self.peo_after_slice_strategy = peo_after_slice_strategy
        if isinstance(base_ordering, str):
            self.base_ordering = qtensor.toolbox.get_ordering_algo(base_ordering)
        else:
            self.base_ordering = base_ordering
        target_tw = kwargs.get('target_tw')
        if target_tw:
            self.max_tw = target_tw

    def _get_max_tw(self):
        if hasattr(self, 'max_tw') and self.max_tw is not None:
            return self.max_tw
        mem = psutil.virtual_memory()
        avail = mem.available
        log.info('Memory available: {}', avail)
        # Cost = 16*2**tw
        # tw = log(cost/16) = log(cost) - 4
        return int(np.log2(avail)) - 4

    def _update_peo_after_slice(self, p_graph, slice_vars):
        if self.peo_after_slice_strategy == 'run-again':
            peo_ints, path = self.base_ordering.get_ordering_ints(p_graph)
        elif self.peo_after_slice_strategy == 'TD-reuse':
            # Remove sliced vars from TD graph. Then, reconstruct peo from this TD
            peo_old = self.peo_ints
            peo_ints = [i for i in peo_old if i not in slice_vars]
            nodes, path = qtensor.utils.get_neighbors_path(p_graph, peo_ints)
            # -- Tree re-peo
            g_components = list(nx.connected_components(p_graph))
            #print(f"# of components: {len(g_components)}, # of nodes total: {p_graph.number_of_nodes()}, # of nodes per component: {[len(c) for c in g_components]}")
            from qtree.graph_model.clique_trees import (
                get_tree_from_peo, get_peo_from_tree)
            tree = get_tree_from_peo(p_graph, peo_ints)
            clique_vertices = []
            # ---- re-create peo from tree
            peo_recreate = []
            components = list(nx.connected_components(tree))
            #print("# of components: ", len(components))
            for subtree in components:
                peo_recreate += get_peo_from_tree(tree.subgraph(subtree).copy(), clique_vertices=clique_vertices)
            # ----
            nodes, path_recreate = qtensor.utils.get_neighbors_path(p_graph, peo_recreate)
            log.info(f"Re-created peo width from tree: {max(path_recreate)}")
            if max(path_recreate) < max(path):
                log.info("Re-created peo is better than old peo. Using new peo.")
                peo_ints = peo_recreate
                path = path_recreate
            # --

        else:
            raise ValueError('Unknown peo_after_slice_strategy: {}'
                             .format(self.peo_after_slice_strategy))

        self.peo_ints = peo_ints
        self.treewidth = max(path)
        log.info('Treewidth after slice: {}', self.treewidth)
        return peo_ints, path

    def _split_graph(self, p_graph, max_tw):
        searcher = GreedyParvars(p_graph)
        while True:
            #nodes, path = utils.get_neighbors_path(graph, peo=peo_ints)
            tw = self.treewidth
            if tw < max_tw:
                log.info(f'Found {len(searcher.result)} parvars: {searcher.result}')
                break
            if self.max_slice is not None:
                if len(searcher.result) > self.max_slice:
                    break
            error = searcher.step()
            pv_cnt = len(searcher.result)
            log.debug('Parvars count: {}. Amps count: {}', pv_cnt, 2**pv_cnt)
            if error:
                log.error('Memory is not enough. Max tw: {}', max_tw)
                raise Exception('Estimated OOM')

            self._update_peo_after_slice(p_graph, searcher.result)

        return self.peo_ints, searcher.result

    def optimize(self, tensor_net):
        peo, tn = super().optimize(tensor_net)
        return peo+self.parallel_vars, self.parallel_vars, tn

    def get_ordering_ints(self, graph, inplace=True):
        p_graph = copy.deepcopy(graph)
        max_tw = self._get_max_tw()
        max_tw = max_tw - self.tw_bias
        log.info('Maximum treewidth: {}', max_tw)

        self.peo_ints, path = self.base_ordering.get_ordering_ints(p_graph)
        self.treewidth = max(path)
        peo, par_vars = self._split_graph(p_graph, max_tw)

        # TODO: move these platform-dependent things
        self.parallel_vars = [
            qtree.optimizer.Var(var,
                                size=graph.nodes[var]['size'],
                                name=graph.nodes[var]['name'])
                              for var in par_vars]
        #log.info('peo {}', self.peo)
        #print('graph nodes', len(graph.nodes))
        #print('pgraph nodes', len(p_graph.nodes))
        # Remove parallel vars from graph
        if hasattr(self, 'graph'):
            for var in par_vars:
                qtree.graph_model.base.remove_node(self.graph, var)
        #self.graph = p_graph
        return peo, [self.treewidth]

class TamakiOptimizer(Optimizer):
    def __init__(self, max_width=None, *args, wait_time=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.wait_time = wait_time
        self.max_width = max_width

    def get_ordering_ints(self, graph, inplace=True):
        peo, tw = qtree.graph_model.peo_calculation.get_upper_bound_peo_pace2017_interactive(
                graph, method="tamaki", max_time=self.wait_time, max_width=self.max_width)
        return peo, [tw]

    def _get_ordering(self, graph, inplace=True):
        node_names = nx.get_node_attributes(graph, 'name')
        node_sizes = nx.get_node_attributes(graph, 'size')
        peo, path = self.get_ordering_ints(graph, inplace=inplace)
        peo = [qtree.optimizer.Var(var, size=node_sizes[var],
                        name=node_names[var])
                    for var in peo]
        self.treewidth = max(path)
        return peo, path

class TamakiExactOptimizer(GreedyOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_ordering(self, graph, inplace=True):
        node_names = nx.get_node_attributes(graph, 'name')
        node_sizes = nx.get_node_attributes(graph, 'size')
        peo, tw = qtree.graph_model.peo_calculation.get_upper_bound_peo_pace2017_interactive(
                graph, method="tamaki_exact", max_time=np.inf)


        peo = [qtree.optimizer.Var(var, size=node_sizes[var],
                        name=node_names[var])
                    for var in peo]
        self.treewidth = tw
        return peo, [tw]

class TreeTrimSplitter(SlicesOptimizer):
    cost_type = 'length'
    def _split_graph(self, p_graph, max_tw):
        peo_ints = self.peo_ints
        tw = self.treewidth
        self._slice_hist = []
        self._slice_hist.append([0, tw, peo_ints])
        log.info('Treewidth: {}', tw)
        log.info('Target treewidth: {}', max_tw)
        result = []
        delta = tw - max_tw
        while delta > 0:
            if hasattr(self, 'par_var_step') and self.par_var_step:
                var_target = self.par_var_step
            else:
                var_target = int((delta)*.2) + 1
            if self.max_slice is not None:
                if len(result) > self.max_slice:
                    break
            # var_target(1) = 1
            # var_target(2) = 2
            # var_target(15) = 12
            # -- relabel

            graph, label_dict = qtree.graph_model.relabel_graph_nodes(
                p_graph, dict(zip(peo_ints, range(len(p_graph.nodes()))))
            )
            if self.free_indices:
                inv_label_dict = {v:k for k,v in label_dict.items()}
                ignore_indices = [inv_label_dict[int(v)] for v in self.free_indices]
            else:
                ignore_indices = []
            #print('ignore_indices', ignore_indices)
            if self.cost_type == 'width':
                par_vars, _ = qtree.graph_model.splitters.split_graph_by_tree_trimming_width(graph, var_target, ignore_indices=ignore_indices)
            else:
                par_vars, _ = qtree.graph_model.splitters.split_graph_by_tree_trimming(graph, var_target, ignore_indices=ignore_indices)
            par_vars = [label_dict[i] for i in par_vars]
            for var in  par_vars:
                log.debug('Remove node {}. Hood size {}', var, utils.n_neighbors(p_graph, var))
                qtree.graph_model.base.remove_node(p_graph, var)
            result += par_vars
            # -- dislabel
            pv_cnt = len(result)
            log.info('Parvars count: {}. Amps count: {}', pv_cnt, 2**pv_cnt)

            peo_ints, path = self._update_peo_after_slice(p_graph, result)
            tw = max(path)
            self._slice_hist.append([pv_cnt, tw, peo_ints])
            delta = tw - max_tw

        return peo_ints, result


class TamakiTrimSlicing(TreeTrimSplitter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_ordering = TamakiOptimizer(*args, **kwargs)


# an alias that makes sense


DefaultOptimizer = GreedyOptimizer
