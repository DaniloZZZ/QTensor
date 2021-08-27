"""
This module implements interface to KaHypar program.
"""
#import qtensor
#from qtensor.optimisation.kahypar_ordering import generate_TN
import kahypar as kahypar
from os.path import join, abspath, dirname

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

modes = ['direct', 'recursive']
objectives = ['cut', 'km1']  
KAHYPAR_PROFILE_DIR = join(abspath(dirname(__file__)), 'config')
#KAHYPAR_PROFILE_DIR = join(abspath(dirname(__file__)))

def set_context(**kwargs):
        mode = modes[int(kwargs.get('mode'))]
        objective = objectives[int(kwargs.get('objective'))]
        K = int(kwargs.get('K'))
        eps = kwargs.get('eps')
        seed = kwargs.get('seed')
        profile_mode = {'direct': 'k', 'recursive': 'r'}[mode]
        profile = f"{objective}_{profile_mode}KaHyPar_sea20.ini"
        context = kahypar.Context()
        context.loadINIconfiguration(join(KAHYPAR_PROFILE_DIR, profile))
        context.setK(K)
        context.setSeed(seed)
        context.setEpsilon(kwargs.get('epsilon', eps * (K - 1))) #imbalance
        context.suppressOutput(True)
        return context

def ka_hg_init(tn): # tn: a dictionary from circ2tn
    h = tn.values()
    l = list({l for word in h for l in word}) #set of unique edges (eg, v_1)
    l.sort()
    nodes = list(range(0, len(l)))
    edges = []
    hyperedge_indices = [0]
    hyperedges = []
    edge_weights = []
    node_weights = []
    for count, value in enumerate(h):
        edges.append([l.index(i) for i in value])
        hyperedges += edges[count]
        hyperedge_indices.append(len(hyperedges))
        edge_weights.append(1)
        
    return nodes, edges, hyperedge_indices, hyperedges, edge_weights, node_weights, l
     
# def num_same_element(a:list, b:list):
#     num = 0
#     for i in a:
#         if (i in b):
#             num = num + 1
#     return num    

def single_partition(tn,**kwargs): 
    K = int(kwargs.get('K'))
    nodes, edges, hyperedge_indices, hyperedges, edge_weights, node_weights, l = ka_hg_init(tn)
    hypergraph = kahypar.Hypergraph(len(nodes), len(edges), hyperedge_indices, \
                                    hyperedges, K, edge_weights, node_weights)
    
    ### Set context
    # context = set_context(K = K, eps = 0.03, mode = 1, objective = 0)    
    context = set_context(**kwargs)
    
    ### perform partition
    kahypar.partition(hypergraph, context)
    partitions_names = [[] for _ in range(K)]
    for i, n in list(enumerate(nodes)):
        partitions_names[hypergraph.blockID(i)].append(n)
        
    return partitions_names, l

def subgraph_partition(tn,**kwargs):
    tn_partite_list = []
    # construct subgraphs
    partitions_names, vertex_list = single_partition(tn,**kwargs) 
    for partite_nodes in partitions_names:
        tn_partite = {}
        if len(partite_nodes) == 0:
            #tn_partite_list.append(tn_partite)
            continue
        # construct a tn for each partite_nodes  
        partite_nodes_name = [vertex_list[t] for t in partite_nodes]
        items = tn.items()
        for (edge, vertex) in items:
            check =  all(item in partite_nodes_name for item in vertex)
            
            # determine an edge in subgraph, needs to test
            if check is True:       
                tn_partite[edge] = vertex 
                continue 
            # else: 
            #     if num_same_element(partite_nodes_name, vertex) >= 2:
            #         tn_partite[edge] = list(set(partite_nodes_name).intersection(set(vertex)))
            #         continue                   
        tn_partite_list.append(tn_partite)
        
    return tn_partite_list
  

def recur_partition(tn,**kwargs):          
    # construct a tree, grow layer by layer
    # order = []
    layer = 0;
    tn_partite_list = [[]]     
    # first partition     
    tn_partite_list[layer] = subgraph_partition(tn,**kwargs)
        
    # grow the tree
    # if the size > K-1, call the partition solver 
    K = int(kwargs.get('K'))
    while max([len(x) for x in tn_partite_list[layer]]) > K-1:
        layer += 1
        tn_partite_list.append([])
        for (count,subgraph) in enumerate(tn_partite_list[layer-1]):
            if subgraph != {}: # important
                tn_partite_list[layer][2*count:2*count] = subgraph_partition(subgraph,**kwargs)
       
        # TODO: Adjust the hyperparameters during the partition
        
    return tn_partite_list


def find_child_ind(subgraph,tn_partite_list, layer):
    # child index is a list  
    ind_last = []
    if subgraph != {} : 
        node_list = list({l for word in list(subgraph.values()) for l in word}) 
        for (count_last,subgraph_last) in enumerate(tn_partite_list[layer+1]):
                    if subgraph_last != {}:
                        node_list_last = list({l for word in list(subgraph_last.values()) for l in word}) 
                        check =  all(item in node_list for item in node_list_last) 
                        if check is True:
                            ind_last.append(count_last)                           
    return ind_last

def find_parent_ind(subgraph,tn_partite_list, layer):
    # parent index is a scalar (int) 
    if subgraph != {} : 
        node_list = list({l for word in list(subgraph.values()) for l in word}) 
        for (count_last,subgraph_last) in enumerate(tn_partite_list[layer-1]):
                    if subgraph_last != {}:
                        node_list_last = list({l for word in list(subgraph_last.values()) for l in word}) 
                        check =  all(item in node_list_last for item in node_list) 
                        if check is True:
                            ind_last=count_last                           
    return ind_last
    
    
def tree2order(tn,tn_partite_list):
    # find the order from top to bottom, arbitary K
    all_edge = list(tn.keys())
    layer_num = len(tn_partite_list) 
    #K = len(tn_partite_list[0])
    order = []
    import copy
    order_tree = copy.deepcopy(tn_partite_list)
    t = 0 # count the temp result in order_tree
    #sub_opt = False # local order search for the subgraph               
    for layer in range(layer_num):
        if layer == 0: 
            # top layer   
            set_last = []
            for subgraph in tn_partite_list[layer]:
                set_last.extend(list(subgraph.keys()))
            result = list(set(all_edge) - set(set_last))
            if result == [] :
                result = [f'temp_{t}']
                t = t + 1
            order.extend(result) 
                
            # layer0 and layer1          
            for (count,subgraph) in enumerate(tn_partite_list[layer]):
                ind_last = find_child_ind(subgraph,tn_partite_list, layer)
                set_last = []
                [set_last.extend(list(tn_partite_list[layer+1][ind].keys())) for ind in ind_last]  
                result = list(set(subgraph) - set(set_last))
                if result == []:
                    result = [f'temp_{t}']
                    t = t + 1
                order[0:0] = result
                order_tree[layer][count] = result
        else:                        
            #Middle layers, need to insert order 
            for (count,subgraph) in enumerate(tn_partite_list[layer]):
                if subgraph != {} : 
                    ind_last = []
                    if layer != layer_num - 1:
                        ind_last = find_child_ind(subgraph,tn_partite_list, layer)
                                               
                    set_last = []
                    [set_last.extend(list(tn_partite_list[layer+1][ind].keys())) for ind in ind_last]  
                    result = list(set(subgraph) - set(set_last))
                    if result == []:
                        result = [f'temp_{t}']
                        t = t + 1
                    partent_ind = find_parent_ind(subgraph,tn_partite_list, layer)
                    parent_set = list(order_tree[layer-1][partent_ind])
                    exist_order = [order.index(x) for x in list(parent_set) if x in order]
                    ind = min(exist_order)
                    order[ind:ind] = result
                    order_tree[layer][count] = result                        
                      
    order = [x for x in order if type(x) != str]
    assert len(order) == len(all_edge)
    # complete the top of order_tree
    set_last1 = set(list(tn_partite_list[0][0].keys()))
    set_last2 = set(list(tn_partite_list[0][1].keys()))  
    result = list(set(all_edge) - set_last1 - set_last2)
    order_tree= [result] + order_tree
    return order,order_tree

def tree2order_old(tn,tn_partite_list):
    # find the order from top to bottom, K=2 only
    all_edge = list(tn.keys())
    layer_num = len(tn_partite_list) 
    order = []
    import copy
    order_tree = copy.deepcopy(tn_partite_list)
    t = 0 # count the temp result in order_tree
    #sub_opt = False # local order search for the subgraph
    for layer in range(layer_num):
        if layer == 0: 
            # top layer          
            set_last1 = set(list(tn_partite_list[layer][0].keys()))
            set_last2 = set(list(tn_partite_list[layer][1].keys()))  
            result = list(set(all_edge) - set_last1 - set_last2)
            if result == [] :
                result = [f'temp_{t}']
                t = t + 1
            order.extend(result) 
            
            # layer0 and layer1
            for (count,subgraph) in enumerate(tn_partite_list[layer]): 
                set_last1 = set(list(tn_partite_list[layer+1][2*count].keys()))
                set_last2 = set(list(tn_partite_list[layer+1][2*count+1].keys()))
                result = list(set(subgraph) - set_last1 - set_last2)
                if result == [] :
                            result = [f'temp_{t}']
                            t = t + 1
                order[0:0]=result
                order_tree[layer][count] = result
        else:                        
            #Middle layers, need to insert order 
            for (count,subgraph) in enumerate(tn_partite_list[layer]):
                if subgraph != {} : 
                    node_list = list({l for word in list(subgraph.values()) for l in word}) 
                    ind_last = []
                    if layer != layer_num - 1: 
                        # find the child node of the subgraph
                        for (count_last,subgraph_last) in enumerate(tn_partite_list[layer+1]):
                            if subgraph_last != {}:
                                node_list_last = list({l for word in list(subgraph_last.values()) for l in word}) 
                                check =  all(item in node_list for item in node_list_last) 
                                if check is True:
                                    ind_last.append(count_last)
                    
                    # if sub_opt is True: 
                    #           # TODO: when the subgraph is small, call other order optimizor)
                    #           #result=(local_search(subgraph))
                    #           find parent set ...
                    
                            
                    if len(ind_last) == 2:    
                        set_last1 = set(list(tn_partite_list[layer+1][ind_last[0]].keys()))
                        set_last2 = set(list(tn_partite_list[layer+1][ind_last[1]].keys()))
                        result = list(set(subgraph) - set_last1 - set_last2)
                        if result == [] :
                            result = [f'temp_{t}']
                            t = t + 1
                            
                        for (count_last,subgraph_last) in enumerate(tn_partite_list[layer-1]):      
                            node_list_last = list({l for word in list(subgraph_last.values()) for l in word}) 
                            check =  all(item in node_list_last for item in node_list)
                            if check is True:
                                partent_ind = count_last
                        #assert len(partent_ind) == 1
                        parent_set = list(order_tree[layer-1][partent_ind])
                        '''
                        while parent_set == []:
                            temp_layer = layer
                            if (partent_ind % 2 == 0):
                                parent_set = list(order_tree[temp_layer-1][partent_ind+1])
                                exist_order = [order.index(x) for x in list(parent_set) if x in order]
                            else:
                                parent_set = list(order_tree[temp_layer-1][partent_ind-1])
                                exist_order = [order.index(x) for x in list(parent_set) if x in order]
                            temp_layer = temp_layer - 1 
                            for (count_last,subgraph_last) in enumerate(tn_partite_list[temp_layer-1]):      
                                node_list_last = list({l for word in list(subgraph_last.values()) for l in word}) 
                                check =  all(item in node_list_last for item in node_list)
                                if check is True:
                                    partent_ind = count_last
                        '''
                        exist_order = [order.index(x) for x in list(parent_set) if x in order]
                        ind = min(exist_order)
                        order[ind:ind]=result
                        order_tree[layer][count] = result

                    if len(ind_last) == 1: 
                        set_last = set(list(tn_partite_list[layer+1][ind_last[0]].keys()))
                        result = list(set(subgraph) - set_last)
                        if result == [] :
                            result = [f'temp_{t}']
                            t = t + 1
                        for (count_last,subgraph_last) in enumerate(tn_partite_list[layer-1]):      
                            node_list_last = list({l for word in list(subgraph_last.values()) for l in word}) 
                            check =  all(item in node_list_last for item in node_list)
                            if check is True:
                                partent_ind = count_last
                        #assert len(partent_ind) == 1
                        parent_set = list(order_tree[layer-1][partent_ind])
                        '''
                        while parent_set == []:
                            temp_layer = layer
                            if (partent_ind % 2 == 0):
                                parent_set = list(order_tree[temp_layer-1][partent_ind+1])
                                exist_order = [order.index(x) for x in list(parent_set) if x in order]
                            else:
                                parent_set = list(order_tree[temp_layer-1][partent_ind-1])
                                exist_order = [order.index(x) for x in list(parent_set) if x in order]
                            temp_layer = temp_layer - 1 
                            for (count_last,subgraph_last) in enumerate(tn_partite_list[temp_layer-1]):      
                                node_list_last = list({l for word in list(subgraph_last.values()) for l in word}) 
                                check =  all(item in node_list_last for item in node_list)
                                if check is True:
                                    partent_ind = count_last
                        '''
                        exist_order = [order.index(x) for x in list(parent_set) if x in order]
                        ind = min(exist_order)
                        order[ind:ind]=result
                    order_tree[layer][count] = result
                    
                    if len(ind_last) == 0:
                        result = list(subgraph)
                        if result == [] :
                            result = [f'temp_{t}']
                            t = t + 1
                        for (count_last,subgraph_last) in enumerate(tn_partite_list[layer-1]):      
                            node_list_last = list({l for word in list(subgraph_last.values()) for l in word}) 
                            check =  all(item in node_list_last for item in node_list)
                            if check is True:
                                partent_ind = count_last
                        #assert len(partent_ind) == 1
                        parent_set = list(order_tree[layer-1][partent_ind])
                        '''
                        while parent_set == []:
                            temp_layer = layer
                            if (partent_ind % 2 == 0):
                                parent_set = list(order_tree[temp_layer-1][partent_ind+1])
                                exist_order = [order.index(x) for x in list(parent_set) if x in order]
                            else:
                                parent_set = list(order_tree[temp_layer-1][partent_ind-1])
                                exist_order = [order.index(x) for x in list(parent_set) if x in order]
                            temp_layer = temp_layer - 1 
                            if temp_layer >= 0:
                                for (count_last,subgraph_last) in enumerate(tn_partite_list[temp_layer-1]):      
                                    node_list_last = list({l for word in list(subgraph_last.values()) for l in word}) 
                                    check =  all(item in node_list_last for item in node_list)
                                    if check is True:
                                        partent_ind = count_last
                            else:
                                continue
                        '''
                        exist_order = [order.index(x) for x in list(parent_set) if x in order]
                        ind = min(exist_order)
                        order[ind:ind]=result
                        order_tree[layer][count] = result
                        
    order = [x for x in order if type(x) != str]
    assert len(order) == len(all_edge)
    # complete the top of order_tree
    set_last1 = set(list(tn_partite_list[0][0].keys()))
    set_last2 = set(list(tn_partite_list[0][1].keys()))  
    result = list(set(all_edge) - set_last1 - set_last2)
    order_tree= [result] + order_tree
    return order,order_tree

def tree2order_old_2(tn,tn_partite_list):
    # find the order from bottom to top
    # correct order_tree, 
    # TODO: there is some bugs in order insertion (find the index of children)
    all_edge = list(tn.keys())
    tn_partite_list = tn_partite_list[::-1]
    import copy
    order_tree = copy.deepcopy(tn_partite_list)
    t = 0 # count the temp result in order_tree
    layer_num = len(tn_partite_list) 
    order = []
    sub_opt = False # local order search for the bottom graph
    for layer in range(layer_num):
        if layer == 0: #bottom layer, append order if there is an edge
            for (count,subgraph) in enumerate(tn_partite_list[layer]):
                if sub_opt is True: 
                    # TODO: when the subgraph is small, call other order optimizor)
                    #order.append(local_search(subgraph))
                    continue
                else:
                    #if subgraph != {} :                
                    result = list(subgraph.keys()) 
                    if result == [] :
                        result = [f'temp_{t}']
                        t = t+1
                    order.extend(result)
                    order_tree[layer][count]=result
        else: 
            #non-bottom layer, need to insert order 
            for (count,subgraph) in enumerate(tn_partite_list[layer]):
                #left_node_empty = 0
                if 1 == 1 :
                    node_list = list({l for word in list(subgraph.values()) for l in word}) 
                    ind_last = []
                    # find the child node of the subgraph
                    for (count_last,subgraph_last) in enumerate(tn_partite_list[layer-1]):
                        if subgraph_last != {}:
                            node_list_last = list({l for word in list(subgraph_last.values()) for l in word}) 
                            check =  all(item in node_list for item in node_list_last) 
                            if check is True:
                                ind_last.append(count_last)
                                
                    if len(ind_last) == 2:    
                        last_set1 = set(list(tn_partite_list[layer-1][ind_last[0]].keys()))
                        last_set2 = set(list(tn_partite_list[layer-1][ind_last[1]].keys()))
                        result = list(set(subgraph) - last_set1 - last_set2)
                        if result == [] :
                            result = [f'temp_{t}']
                            t = t + 1
                        # for (count_last,subgraph_last) in enumerate(tn_partite_list[layer-1]):      
                        #     node_list_last = list({l for word in list(subgraph_last.values()) for l in word}) 
                        #     check =  all(item in node_list for item in node_list_last)
                        #     if check is True:
                        #         child_ind = count_last
                        child_set1 = list(order_tree[layer-1][ind_last[0]])
                        child_set2 = list(order_tree[layer-1][ind_last[1]])
                        
                        exist_order = [order.index(x) for x in list(child_set1) if x in order] + \
                            [order.index(x) for x in list(child_set2) if x in order]
                        ind = max(exist_order)+1
                        order[ind:ind]=result
                        order_tree[layer][count]=result
                    
                    ### there is a single node partition in the subgraph    
                    if len(ind_last) == 1: 
                        last_set = set(list(tn_partite_list[layer-1][ind_last[0]].keys()))
                        result = list(set(subgraph) - last_set)
                        if result == [] :
                            result = [f'temp_{t}']
                            t = t + 1
                        # for (count_last,subgraph_last) in enumerate(tn_partite_list[layer-1]):      
                        #     node_list_last = list({l for word in list(subgraph_last.values()) for l in word}) 
                        #     check =  all(item in node_list for item in node_list_last)
                        #     if check is True:
                        #         child_ind = count_last
                        child_set = list(order_tree[layer-1][ind_last[0]])
                        exist_order = [order.index(x) for x in list(child_set) if x in order]
                        ind = max(exist_order)+1
                        order[ind:ind]=result
                        order_tree[layer][count]=result
                            
                    ### there are two single node partition in the subgraph
                    if len(ind_last) == 0:  
                        result = list(subgraph)
                        if result == [] :
                            result = [f'temp_{t}']
                            t = t + 1
                        for (count_last,subgraph_last) in enumerate(tn_partite_list[layer-1]):      
                            node_list_last = list({l for word in list(subgraph_last.values()) for l in word}) 
                            check =  all(item in node_list for item in node_list_last)
                            if check is True:
                                child_ind = count_last
                        child_set = list(order_tree[layer-1][child_ind])
                        
                        exist_order = [order.index(x) for x in list(child_set) if x in order]
                        ind = max(exist_order)+1
                        order[ind:ind]=result
                        order_tree[layer][count] = result
                        
                '''        
                        #count = tn_partite_list[layer].index(subgraph)
                        if count % 2 == 0: # left node
                            # find the contracted edge of its paired node in the same layer
                            if len(tn_partite_list[layer][count+1].keys()) < 2:
                                order.extend(result) 
                                order_tree[layer][count]=result
                                # both left and right subgraphs are empty, new start
                            else:
                                left_node_empty = 1
                                empty_count = count
                                left_node_buffer = result
                                order_tree[layer][count]=result
                                # follow the order of the right-paired subgraph   
                        else: # right node
                            if len(tn_partite_list[layer][count-1].keys()) < 2:
                                    order.extend(result)
                                    order_tree[layer][count]=result
                                # both left and right subgraphs are empty, new start 
                            else:
                                exist_order = [order.index(x) for x in list(tn_partite_list[layer][count-1]) if x in order]
                                ind = max(exist_order)+1
                                order[ind:ind] = result
                                order_tree[layer][count]=result
                                # follow the order of the left-paired subgraph
                if left_node_empty != 0:
                    exist_order = [order.index(x) for x in list(tn_partite_list[layer][empty_count+1]) if x in order]
                    ind = max(exist_order)+1
                    order[ind:ind] = left_node_buffer
            '''           
                        
            if layer == layer_num - 1:
                set_last1 = set(list(tn_partite_list[layer][0].keys()))
                set_last2 = set(list(tn_partite_list[layer][1].keys()))  
                result = list(set(all_edge) - set_last1 - set_last2)
                if result == [] :
                            result = [f'temp_{t}']
                            t = t + 1
                order.extend(result) 
                order_tree.append(result)
                
    order = [x for x in order if type(x) != str]                
    assert len(order) == len(all_edge)
    order_tree = order_tree[::-1]
    
    return order, order_tree
