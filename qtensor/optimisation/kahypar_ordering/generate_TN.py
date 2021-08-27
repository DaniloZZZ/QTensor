from typing import List, Dict, Hashable, Iterable
import qtree
import qtensor


Edge = Iterable[Hashable]
Hypergraph = Dict[Hashable, Edge]
# # self = hypergraph
# verts = set(sum(self.values(), []))
# num_edges = len(self)

def dual_hg(hg: Hypergraph) -> Hypergraph:
    dual = {}
    for iedge, edge in hg.items():
        for vert in edge:
            if dual.get(vert) is None:
                dual[vert] = []
            dual[vert].append(iedge)
    return dual

def circ2tn(circuit: List[qtree.operators.Gate]):
    tn = qtensor.optimisation.QtreeTensorNet.from_qtree_gates(circuit)
    dual_tn = { str(hex(id(t))):t.indices for t in tn.tensors }
    TN = dual_hg(dual_tn)
    return TN

