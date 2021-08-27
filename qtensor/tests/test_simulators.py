import qtensor
from qtensor import CirqQAOAComposer, QtreeQAOAComposer
from qtensor import QAOAQtreeSimulator
from qtensor.Simulate import CirqSimulator, QtreeSimulator
from qtensor.FeynmanSimulator import FeynmanSimulator
import numpy as np
import networkx as nx

np.random.seed(42)


def get_test_problem(n=14, p=2, d=3):
    w = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]])
    G = nx.from_numpy_matrix(w)

    G = nx.random_regular_graph(d, n)
    gamma, beta = [np.pi/3]*p, [np.pi/2]*p
    return G, gamma, beta

def test_qtree():
    G, gamma, beta = get_test_problem()

    composer = QtreeQAOAComposer(
        graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()

    print(composer.circuit)
    sim = QtreeSimulator()
    result = sim.simulate(composer.circuit)
    print(result)
    qtree_amp = result

    composer = CirqQAOAComposer(
        graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()

    print(composer.circuit)
    sim = CirqSimulator()
    result = sim.simulate(composer.circuit)
    print(result)
    final_cirq = result.final_state_vector
    assert np.allclose(final_cirq[0], qtree_amp)

    assert result


def test_parallel_batched():
    G, gamma, beta = get_test_problem(14, 3, d=4)
    batch_vars = 3

    composer = QtreeQAOAComposer(
        graph=G, gamma=gamma, beta=beta)
    composer.ansatz_state()

    sim = QtreeSimulator()
    amp = sim.simulate(composer.circuit)
    amps = sim.simulate_batch(composer.circuit, batch_vars=2)
    print('ordinary qtree amp', amp)
    print('ordinary qtree 2 amps', amps)
    assert abs( amp - amps[0]) < 1e-6

    sim = FeynmanSimulator()
    result = sim.simulate(composer.circuit, batch_vars=batch_vars, tw_bias=7)
    print(result)

    batch_amps = 2**batch_vars
    assert len(result) == batch_amps
    assert abs( amp - result[0]) < 1e-6


def test_qtree_energy():
    G, gamma, beta = get_test_problem(16, 2, d=3)

    sim = QAOAQtreeSimulator(QtreeQAOAComposer)
    E = sim.energy_expectation(
        G=G, gamma=gamma, beta=beta)

    print('Energy', E)
    assert np.imag(E)<1e-6

    E = np.real(E)

    Ed = G.number_of_edges()
    C = (Ed - E)/2

    print("Edges", Ed)
    print("Cost", C)
    assert E

if __name__ == "__main__":
    #test_qtree_energy()
    test_parallel_batched()
