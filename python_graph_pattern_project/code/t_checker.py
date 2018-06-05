import numpy as np
from config import Config
from DirectedWeightedGraph import DWGraph
from typing import Tuple, Dict, Set, Union
from copy import deepcopy
import pickle

CONFIG = Config.get_config("default")


def inverse_f(f):
    f_inv = {}
    for k, val_set in f.items():
        for v in val_set:
            f_inv.setdefault(v, set()).add(k)
    return f_inv


def labeling(g: DWGraph):
    nodes = [n for n in g.nodes.keys()]
    labels = np.array(nodes)
    inv_labels = {}
    for i, k in enumerate(nodes):
        inv_labels[k] = i
    return labels, inv_labels


def f_edge(f_inv: Dict[int, Set[int]], g: DWGraph, g0: DWGraph, edges_matching: Union[DWGraph.EdgesMatchingType, None]):
    g0_labeling, g0_inv_labels = labeling(g0)
    g0_n = g0_labeling.shape[0]
    g0_labels = g0.get_labels()
    g_labels = g.get_labels()
    mtx = np.ndarray((g0_n, g0_n), dtype=object)

    for i in range(g0_n):
        for j in range(g0_n):
            mtx[i, j] = set()

    # Init matrix
    for i in range(g0_n):
        q0 = g0_labeling[i]
        for q in g0[q0].keys():
            j = g0_inv_labels[q]
            for v0 in f_inv[q0]:
                for v in f_inv[q]:
                    # Check if the edges (q0, q) and (v0, v) have the same type.
                    if v in g[v0]:
                        if edges_matching is None:
                            flag = True
                        else:
                            g_e = (g_labels[v0], g_labels[v], g.get_max_weight(v0, v))
                            g0_e = (g0_labels[q0], g0_labels[q], g0.get_max_weight(q0, q))
                            flag = edges_matching(g_e, g0_e)

                        if flag:
                            mtx[i, j].add(((q0, v0), (q, v)))

    return mtx


def elements_mult(a: Set[Tuple[Tuple[int, int]]], b: Set[Tuple[Tuple[int, int]]]):
    r = set()
    for pa in a:
        for pb in b:
            flag = True
            for qa, va in pa:
                for qb, vb in pb:
                    if (qa == qb and va != vb) or (qa != qb and va == vb):
                        flag = False
                        break

                if not flag:
                    break

            if flag:
                r.add(pa + pb[1:])

    return r


def elements_intersection(a: Set[Tuple[Tuple[int, int]]], b: Set[Tuple[Tuple[int, int]]]):
    if not a:
        return b
    elif not b:
        return a
    _a_ = set([l[0][1] for l in a])
    _b_ = set([l[0][1] for l in b])
    _c_ = _a_.intersection(_b_)

    a_e = set([l[-1][1] for l in a])
    b_e = set([l[-1][1] for l in b])
    c_e = a_e.intersection(b_e)

    r = set()
    for l in a:
        if l[0][1] in _c_ and l[-1][1] in c_e:
            r.add(l)

    for l in b:
        if l[0][1] in _c_ and l[-1][1] in c_e:
            r.add(l)

    return r


def mtx_power2(mtx_a, n):
    r = deepcopy(mtx_a)
    for i in range(n):
        r = mtx_mult(r, r)

    return r


def mtx_mult(mtx_a, mtx_b):
    print("MULT")
    n = mtx_a.shape[0]
    mtx = np.ndarray((n, n), dtype=object)
    for i in range(n):
        print("i = {}".format(i))
        for j in range(n):
            r = set()
            for k in range(n):
                a = mtx_a[i, k]
                b = mtx_b[k, j]
                c = elements_mult(a, b)
                r = elements_intersection(r, c)
            mtx[i, j] = r
    return mtx


def main():
    c = CONFIG
    with open(c.output_t, 'r') as f:
        o = eval(f.read())

    # in_list = [k in v for k, v in o.items()]
    # print(in_list)
    f_inv = inverse_f(o)
    print(len(o.keys()), len(f_inv.keys()))
    print("f: ", o)
    print("f-1:", f_inv)

    # Load graph
    g = DWGraph.from_file_edge_list(c.background_file_path,
                                    is_weighted=c.background_weighted,
                                    make_undirected=not c.background_keep_directed,
                                    skip_first_line=c.background_skip_header)
    # Add labels to graph
    if c.background_labeling_function is not None:
        g.add_labels_by_identifiers(c.background_labeling_function)
    elif c.background_labeling_file is not None:
        g.add_labels_from_file(c.background_labeling_file)
    else:
        g.add_default_labels()

    # Load graph
    g0 = DWGraph.from_file_edge_list(c.template_file_path,
                                     is_weighted=c.template_weighted,
                                     make_undirected=not c.template_keep_directed,
                                     skip_first_line=c.template_skip_header)
    # Add labels to graph
    if c.template_labeling_function is not None:
        g0.add_labels_by_identifiers(c.template_labeling_function)
    elif c.template_labeling_file is not None:
        g0.add_labels_from_file(c.template_labeling_file)
    else:
        g0.add_default_labels()

    print(labeling(g))

    fed = f_edge(f_inv, g, g0, None)
    nf = mtx_mult(fed, fed)

    lb, lb_inv = labeling(g0)
    print(nf[lb_inv[1], lb_inv[3]])

    print(len(nf[lb_inv[1], lb_inv[3]]))

    rr = mtx_power2(fed, 10)
    with open('../data/mtx.pickle', 'wb') as pick_f:
        pickle.dump(rr, pick_f)
    print(rr)


if __name__ == '__main__':
    main()
