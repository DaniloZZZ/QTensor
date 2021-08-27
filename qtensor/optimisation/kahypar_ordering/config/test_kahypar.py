import os
import kahypar as kahypar

    
num_nodes = 7
num_nets = 4

hyperedge_indices = [0,2,6,9,12] #Starting indices for each hyperedge
hyperedges = [0,2,0,1,3,4,3,4,6,2,5,6] #Vector containing all hyperedges
# 0,2
# 0,1,3,4
# 3,4,6
# 2,5,6
# http://glaros.dtc.umn.edu/gkhome/metis/hmetis/download manual page 14
# list of set, set is py obj,  each element is ind 

node_weights = [1,2,3,4,5,6,7] # Weights of all hypernodes
edge_weights = [11,22,33,44] # Weights of all hyperedges

k = 2

hypergraph = kahypar.Hypergraph(num_nodes, num_nets, hyperedge_indices, hyperedges, k, edge_weights, node_weights)

context = kahypar.Context()
context.loadINIconfiguration("/Users/zichanghe/Desktop/kahypar/config/cut_rKaHyPar_sea20.ini")

context.setK(k)
context.setEpsilon(0.03)

kahypar.partition(hypergraph, context)