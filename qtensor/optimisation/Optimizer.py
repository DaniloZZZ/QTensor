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
    def optimize(self, tensor_net):
        raise NotImplementedError

class WithoutOptimizer(Optimizer):

    def optimize(self, tensor_net):
        line_graph = tensor_net.get_line_graph()
        free_vars = tensor_net.free_vars
        ignored_vars = tensor_net.ket_vars + tensor_net. bra_vars
        graph = line_graph


        peo = sorted([int(v) for v in graph.nodes()])
        # magic line
        peo = list(reversed(peo))
        _, path = utils.get_neighbors_path(graph, peo)
        self.treewidth = max(path)
        self.peo_ints = peo

        peo = [qtree.optimizer.Var(var, size=graph.nodes[var]['size'],
                        name=graph.nodes[var]['name'])
                    for var in peo]
        if free_vars:
            peo = qtree.graph_model.get_equivalent_peo(graph, peo, free_vars)

        peo = ignored_vars + peo
        self.peo = peo
        self.graph = graph
        self.ignored_vars = ignored_vars
        return peo, tensor_net


# TODO: rename to greedy
class GreedyOptimizer(Optimizer):
    def _get_ordering_ints(self, graph, free_vars=[]):
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

    def optimize(self, tensor_net):
        graph = tensor_net.get_line_graph()
        free_vars = tensor_net.free_vars
        ignored_vars = tensor_net.ket_vars + tensor_net.bra_vars

        if free_vars:
            # It's more efficient to find ordering in-place to avoid copying
            # We'll need the copy of a graph only if we have free_vars
            graph = qtree.graph_model.make_clique_on(graph, free_vars)
            graph_copy = copy.deepcopy(graph)
            self.graph = graph_copy

        peo, path = self._get_ordering(graph, inplace=True)
        self.treewidth = max(path)
        self.peo_ints = [int(x) for x in peo]

        if free_vars:
            peo = qtree.graph_model.get_equivalent_peo(self.graph, peo, free_vars)

        peo = ignored_vars + peo
        self.peo = peo
        self.ignored_vars = ignored_vars
        return peo, tensor_net

###################################################################
class KahyparOptimizer(Optimizer):
   
    def _get_kahyper_kwarge(self):
        # initial hyperparameters: to tune 
        # different from tuning during the recur_partition
        kwargs = {'K': 2, 'eps': 0.1, 'seed': 2021, 'mode':0, 'objective':0} 
        return kwargs
        
    def optimize(self, tensor_net, tn):
        
        #tensor_net=qtensor.optimisation.QtreeTensorNet.from_qtree_gates(circ)
        #free_vars = tensor_net.free_vars
        ignored_vars = tensor_net.ket_vars + tensor_net.bra_vars
            
        kwargs = self._get_kahyper_kwarge()
        #from qtensor.optimisation.kahypar_ordering import generate_TN
        #tn = generate_TN.circ2tn(circ)
        
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

class SlicesOptimizer(GreedyOptimizer):

    def __init__(self, tw_bias=2, max_tw=None, max_slice=None, **kwargs):
        self.tw_bias = tw_bias
        self.max_tw = max_tw
        self.max_slice = max_slice
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
        return np.int(np.log2(avail)) - 4

    def _split_graph(self, p_graph, max_tw):
        searcher = GreedyParvars(p_graph)
        peo_ints, path = self._get_ordering_ints(p_graph)
        while True:
            #nodes, path = utils.get_neighbors_path(graph, peo=peo_ints)
            tw = self.treewidth
            log.info('Treewidth: {}', tw)
            if tw < max_tw:
                log.info('Found parvars: {}', searcher.result)
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

            peo_ints, path = self._get_ordering_ints(p_graph)
            self.treewidth = max(path)

        return peo_ints, searcher.result

    def optimize(self, tensor_net):
        peo, _ = super().optimize(tensor_net)
        try:
            graph = self.graph
        except AttributeError:
            graph = tensor_net.get_line_graph()

        p_graph = graph.copy()
        max_tw = self._get_max_tw()
        log.info('Maximum treewidth: {}', max_tw)
        max_tw = max_tw - self.tw_bias

        peo, par_vars = self._split_graph(p_graph, max_tw)

        # TODO: move these platform-dependent things
        self.parallel_vars = [
            qtree.optimizer.Var(var,
                                size=graph.nodes[var]['size'],
                                name=graph.nodes[var]['name'])
                              for var in par_vars]
        peo = [qtree.optimizer.Var(var, size=graph.nodes[var]['size'],
                        name=graph.nodes[var]['name'])
                    for var in peo]
        if tensor_net.free_vars:
            peo = qtree.graph_model.get_equivalent_peo(p_graph, peo, tensor_net.free_vars)

        self.peo = self.ignored_vars + peo + self.parallel_vars 
        #log.info('peo {}', self.peo)
        return self.peo, self.parallel_vars, tensor_net

class TamakiOptimizer(GreedyOptimizer):
    def __init__(self, *args, wait_time=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.wait_time = wait_time

    def _get_ordering(self, graph, inplace=True):
        node_names = nx.get_node_attributes(graph, 'name')
        node_sizes = nx.get_node_attributes(graph, 'size')
        peo, tw = qtree.graph_model.peo_calculation.get_upper_bound_peo_pace2017_interactive(
                graph, method="tamaki", max_time=self.wait_time)


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
        self._slice_hist.append([0, tw])
        log.info('Treewidth: {}', tw)
        log.info('Target treewidth: {}', max_tw)
        result = []
        delta = tw - max_tw
        while delta > 0:
            if hasattr(self, 'par_var_step') and self.par_var_step:
                var_target = self.par_var_step
            else:
                var_target = int((delta)*.2) + 1
            # var_target(1) = 1
            # var_target(2) = 2
            # var_target(15) = 12
            # -- relabel

            graph, label_dict = qtree.graph_model.relabel_graph_nodes(
                p_graph, dict(zip(peo_ints, range(len(p_graph.nodes()))))
            )
            if self.cost_type == 'width':
                par_vars, _ = qtree.graph_model.splitters.split_graph_by_tree_trimming_width(graph, var_target)
            else:
                par_vars, _ = qtree.graph_model.splitters.split_graph_by_tree_trimming(graph, var_target)
            par_vars = [label_dict[i] for i in par_vars]
            for var in  par_vars:
                log.debug('Remove node {}. Hood size {}', var, utils.n_neighbors(p_graph, var))
                qtree.graph_model.base.remove_node(p_graph, var)
            result += par_vars
            # -- dislabel
            pv_cnt = len(result)
            log.info('Parvars count: {}. Amps count: {}', pv_cnt, 2**pv_cnt)

            peo_ints, path = self._get_ordering_ints(p_graph)
            tw = max(path)
            log.info('Treewidth: {}', tw)
            self._slice_hist.append([pv_cnt, tw])

            delta = tw - max_tw
            self.treewidth = tw


        return peo_ints, result


class TamakiTrimSlicing(TamakiOptimizer, TreeTrimSplitter):
    pass


# an alias that makes sense


DefaultOptimizer = GreedyOptimizer
