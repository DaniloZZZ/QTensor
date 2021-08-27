#!/usr/bin/env python3
import qtensor
from qtensor import CirqQAOAComposer, QtreeQAOAComposer
from qtensor import QAOAQtreeSimulator
from qtensor.optimisation.Optimizer import GreedyOptimizer, TamakiOptimizer, KahyparOptimizer, TamakiTrimSlicing, TreeTrimSplitter
from qtensor.optimisation.Optimizer import SlicesOptimizer
from qtensor.optimisation.TensorNet import QtreeTensorNet
from qtensor.FeynmanSimulator import FeynmanSimulator
import numpy as np
import matplotlib.pyplot as plt 
import math
from qtensor import utils
import pytest
from qtensor.tests import get_test_problem
from qtensor import QAOAQtreeSimulator
from qtensor.Simulate import CirqSimulator, QtreeSimulator

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
np.random.seed(2021)
 
def generate_problem(N,p):
        ##
        G, gamma, beta = get_test_problem(N, p, d=3)
        composer = QtreeQAOAComposer(graph=G, gamma=gamma, beta=beta)
        composer.ansatz_state()
        #composer.energy_expectation_lightcone(list(G.edges())[0])
        
        ##
        #g = nx.random_regular_graph(3, N) # more complicated structure
        #comp = qtensor.DefaultQAOAComposer(g, gamma=[1]*p, beta=[2]*p)
        #comp.ansatz_state()
        
        tn = QtreeTensorNet.from_qtree_gates(composer.circuit)  
        return composer, tn
    
def get_tw_costs_kahypar(composer, tn):
    ### Kahypar
    kahypar_opt = KahyparOptimizer()
    from qtensor.optimisation.kahypar_ordering import generate_TN
    hypar_tn = generate_TN.circ2tn(composer.circuit)
    
    with timing() as t_kahypar:
        kahypar_peo, _ = kahypar_opt.optimize(tn, hypar_tn)
    kahypar_tw = kahypar_opt.treewidth   
    kahypar_mems, kahypar_flops = tn.simulation_cost(kahypar_peo)
    return t_kahypar.result, kahypar_tw, kahypar_mems, kahypar_flops
    
def get_tw_costs_greedy(tn):
    ### Greedy
    greedy_opt = GreedyOptimizer()
    with timing() as t_greedy:
        greedy_peo, _ = greedy_opt.optimize(tn)
    greedy_tw = greedy_opt.treewidth   
    greedy_mems, greedy_flops = tn.simulation_cost(greedy_peo)
    return t_greedy.result, greedy_tw, greedy_mems, greedy_flops

def get_tw_costs_tamaki(tn, wait_time, max_tw=64):
    ### Tamaki        
    tamaki_opt = TamakiOptimizer(wait_time = wait_time) # time to run tamaki, in seconds
    tamaki_opt.max_tw = max_tw
    with timing() as t_tamaki:
        tamaki_peo, _ = tamaki_opt.optimize(tn)
    tamaki_tw = tamaki_opt.treewidth
    tamaki_time = t_tamaki.result
    tamaki_mems, tamaki_flops = tn.simulation_cost(tamaki_peo)
    return tamaki_time, tamaki_tw, tamaki_mems, tamaki_flops

def get_tw_costs_random(tn,peo):
    ### Random
    rand_tw_list = []
    with timing() as t_random:
        for _ in range(1):
            rand_peo=np.random.permutation(peo)
            nodes, ngh = utils.get_neighbors_path(tn.get_line_graph(), rand_peo)
            rand_tw_list.append(max(ngh))
    random_tw = min(rand_tw_list)
    #rand_mems, rand_flops = tn.simulation_cost(rand_peo)
    return t_random.result, random_tw
    
def test_cost_estimation_N(): 
    ### Different N
    kahypar_result = []; tamaki_result=[]; greedy_result=[]
    N_list = [20,40,60,80,100,120]
    for N in N_list:
        p = 3 
        composer, tn = generate_problem(N,p)
        kahypar_str = get_tw_costs_kahypar(composer, tn)
        tamaki_str=[];  wait_time_list = [10,60,150]
        for wait_time in wait_time_list:
            tamaki_str.append(get_tw_costs_tamaki(tn, wait_time))
        greedy_str = get_tw_costs_greedy(tn)
        
        print(f'Problem size N={N}, p={p}, d=3')
        print('--Order Search Time--')
        print(f'Kahypa: {kahypar_str[0]}') 
        #print(f'Random: {t_random.result}') 
        for (count,wait_time) in enumerate(wait_time_list):
            print(f'Tamaki ({wait_time}): {tamaki_str[count][0]}')
        print(f'Greedy: {greedy_str[0]}')
        print()
        
        print('--Width--')
        print(f'Partition tw: {kahypar_str[1]}') 
        #print(f'Random tw: {min(rand_tw_list)}') 
        for (count,wait_time) in enumerate(wait_time_list):
            print(f'Tamaki ({wait_time}) tw: {tamaki_str[count][1]}')
        print(f'Greedy tw: {greedy_str[1]}')
        print()
        
        print('--Log Max memory--')
        print(f'Partition mem: {math.log2(max(kahypar_str[2]))}') 
        for (count,wait_time) in enumerate(wait_time_list):
            print(f'Tamaki ({wait_time}) mem: {math.log2(max(tamaki_str[count][2]))}')
        print(f'Greedy mem: {math.log2(max(greedy_str[2]))}')
        print()
        
        print('--Log Total flops--')
        print(f'Partition flops: {math.log2(sum(kahypar_str[3]))}') 
        for (count,wait_time) in enumerate(wait_time_list):
            print(f'Tamaki ({wait_time}) flops: {math.log2(sum(tamaki_str[count][3]))}')
        print(f'Greedy flops: {math.log2(sum(greedy_str[3]))}')
        print()
        
        kahypar_result.append(kahypar_str)
        tamaki_result.append(tamaki_str)
        greedy_result.append(greedy_str)
        
    ### Plot figure
    tamaki_result = list(map(list, zip(*tamaki_result))) #important
    f, axs = plt.subplots(2, 2, figsize=(10, 10))
    plt.sca(axs[0][0])
    plt.plot(N_list,[math.log2(row[0]) for row in kahypar_result])
    for (count,_) in enumerate(wait_time_list):
        plt.plot(N_list,[math.log2(row[0]) for row in tamaki_result[count]])
    plt.plot(N_list,[math.log2(row[0]) for row in greedy_result])
    plt.title('Log Order Search Time')
    
    plt.sca(axs[0][1])
    plt.plot(N_list,[math.log2(row[1]) for row in kahypar_result]) 
    for (count,_) in enumerate(wait_time_list):
        plt.plot(N_list,[math.log2(row[1]) for row in tamaki_result[count]])
    plt.plot(N_list,[math.log2(row[1]) for row in greedy_result])
    plt.title('Log Width')
    
    plt.sca(axs[1][0])
    mem_temp=[row[2] for row in kahypar_result]
    plt.plot(N_list,[math.log2(max(x)) for x in mem_temp]) 
    for (count,_) in enumerate(wait_time_list):
        mem_temp=[row[2] for row in tamaki_result[count]]
        plt.plot(N_list,[math.log2(max(x)) for x in mem_temp])
    mem_temp=[row[2] for row in greedy_result]
    plt.plot(N_list,[math.log2(max(x)) for x in mem_temp]) 
    plt.title('Log Max memory')
    
    plt.sca(axs[1][1])
    flop_temp=[row[3] for row in kahypar_result]
    plt.plot(N_list,[math.log2(sum(x)) for x in flop_temp], label = 'Kahyper')
    for (count,wait_time) in enumerate(wait_time_list):
        flop_temp=[row[3] for row in tamaki_result[count]]
        plt.plot(N_list,[math.log2(max(x)) for x in flop_temp], label = 'Tamaki (%s)' % wait_time)
    flop_temp=[row[3] for row in greedy_result]
    plt.plot(N_list,[math.log2(max(x)) for x in flop_temp], label = 'Greedy')
    plt.title('Log Total flops')
    
    plt.legend()
    plt.show() # shows a plot if run from terminal on machine with GUI on
    
def test_cost_estimation_p(): 
    ### Different p
    kahypar_result = []; tamaki_result=[]; greedy_result=[]
    p_list = [2,4,6,8,10]
    for p in p_list:
        N = 60
        composer, tn = generate_problem(N,p)
        kahypar_str = get_tw_costs_kahypar(composer, tn)
        tamaki_str=[];  wait_time_list = [10,60,150]
        for wait_time in wait_time_list:
            tamaki_str.append(get_tw_costs_tamaki(tn, wait_time))
        greedy_str = get_tw_costs_greedy(tn)
        
        print(f'Problem size N={N}, p={p}, d=3')
        print('--Order Search Time--')
        print(f'Kahypa: {kahypar_str[0]}') 
        #print(f'Random: {t_random.result}') 
        for (count,wait_time) in enumerate(wait_time_list):
            print(f'Tamaki ({wait_time}): {tamaki_str[count][0]}')
        print(f'Greedy: {greedy_str[0]}')
        print()
        
        print('--Width--')
        print(f'Partition tw: {kahypar_str[1]}') 
        #print(f'Random tw: {min(rand_tw_list)}') 
        for (count,wait_time) in enumerate(wait_time_list):
            print(f'Tamaki ({wait_time}) tw: {tamaki_str[count][1]}')
        print(f'Greedy tw: {greedy_str[1]}')
        print()
        
        print('--Log Max memory--')
        print(f'Partition mem: {math.log2(max(kahypar_str[2]))}') 
        for (count,wait_time) in enumerate(wait_time_list):
            print(f'Tamaki ({wait_time}) mem: {math.log2(max(tamaki_str[count][2]))}')
        print(f'Greedy mem: {math.log2(max(greedy_str[2]))}')
        print()
        
        print('--Log Total flops--')
        print(f'Partition flops: {math.log2(sum(kahypar_str[3]))}') 
        for (count,wait_time) in enumerate(wait_time_list):
            print(f'Tamaki ({wait_time}) flops: {math.log2(sum(tamaki_str[count][3]))}')
        print(f'Greedy flops: {math.log2(sum(greedy_str[3]))}')
        print()
        
        kahypar_result.append(kahypar_str)
        tamaki_result.append(tamaki_str)
        greedy_result.append(greedy_str)
        
    ### Plot figure
    tamaki_result = list(map(list, zip(*tamaki_result))) #important
    f, axs = plt.subplots(2, 2, figsize=(10, 10))
    plt.sca(axs[0][0])
    plt.plot(p_list,[math.log2(row[0]) for row in kahypar_result])
    for (count,_) in enumerate(wait_time_list):
        plt.plot(p_list,[math.log2(row[0]) for row in tamaki_result[count]])
    plt.plot(p_list,[math.log2(row[0]) for row in greedy_result])
    plt.title('Log Order Search Time')
    
    plt.sca(axs[0][1])
    plt.plot(p_list,[math.log2(row[1]) for row in kahypar_result]) 
    for (count,_) in enumerate(wait_time_list):
        plt.plot(p_list,[math.log2(row[1]) for row in tamaki_result[count]])
    plt.plot(p_list,[math.log2(row[1]) for row in greedy_result])
    plt.title('Log Width')
    
    plt.sca(axs[1][0])
    mem_temp=[row[2] for row in kahypar_result]
    plt.plot(p_list,[math.log2(max(x)) for x in mem_temp]) 
    for (count,_) in enumerate(wait_time_list):
        mem_temp=[row[2] for row in tamaki_result[count]]
        plt.plot(p_list,[math.log2(max(x)) for x in mem_temp])
    mem_temp=[row[2] for row in greedy_result]
    plt.plot(p_list,[math.log2(max(x)) for x in mem_temp]) 
    plt.title('Log Max memory')
    
    plt.sca(axs[1][1])
    flop_temp=[row[3] for row in kahypar_result]
    plt.plot(p_list,[math.log2(sum(x)) for x in flop_temp], label = 'Kahyper')
    for (count,wait_time) in enumerate(wait_time_list):
        flop_temp=[row[3] for row in tamaki_result[count]]
        plt.plot(p_list,[math.log2(max(x)) for x in flop_temp], label = 'Tamaki (%s)' % wait_time)
    flop_temp=[row[3] for row in greedy_result]
    plt.plot(p_list,[math.log2(max(x)) for x in flop_temp], label = 'Greedy')
    plt.title('Log Total flops')
    
    plt.legend()
    plt.show() # shows a plot if run from terminal on machine with GUI on
            
def test_qtree():
    N = 20 
    p = 4
    composer, tn = generate_problem(N,p)
    
    ###
    optimizer=TamakiOptimizer(wait_time = 60)
    sim = QtreeSimulator(optimizer = optimizer)
    with timing() as t_tamaki:
        result_tamaki = sim.simulate(composer.circuit)
    
    optimizer=GreedyOptimizer()
    sim = QtreeSimulator(optimizer = optimizer)
    with timing() as t_greedy:
        result_greedy = sim.simulate(composer.circuit)
    
    optimizer=KahyparOptimizer()
    sim = QtreeSimulator(optimizer = optimizer)
    from qtensor.optimisation.kahypar_ordering import generate_TN
    setattr(sim, 'hypar_tn', generate_TN.circ2tn(composer.circuit)) 
    with timing() as t_kahypar:
        result_kahypar = sim.simulate(composer.circuit)
        
    assert np.allclose(result_greedy, result_kahypar)
    
    print('--Simulation Time--')
    print(f'Tamaki: {t_tamaki.result}')  
    print(f'Greedy: {t_greedy.result}')
    print(f'Kahypar: {t_kahypar.result}')  
        
    
if __name__ == '__main__':
    #test_cost_estimation_N()
    #test_cost_estimation_p()
    test_qtree()