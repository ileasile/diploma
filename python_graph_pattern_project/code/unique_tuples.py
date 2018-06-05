from DirectedWeightedGraph import DWGraph
from typing import List, Iterable, Set, Tuple
import numpy as np


def unique_tuples(g: DWGraph, g0_index: List[int], g0: DWGraph,
                  edges_matching, sets: List[Set[int]]) -> Iterable[Tuple[int]]:
    if len(sets) == 0:
        return ()

    u = set()
    for s in sets:
        u = u.union(s)
    u = list(u)

    n = len(sets)
    m = len(u)

    um = np.zeros((n, m), bool)

    for i in range(n):
        for j in range(m):
            um[i, j] = u[j] in sets[i]

    for t in unique_tuples_mtx(g, u, g0, g0_index, edges_matching, um):
        yield tuple([u[j] for j in t])


def check_mtx(a, z):
    n = np.count_nonzero(np.count_nonzero(a, axis=0))
    k = np.count_nonzero(np.count_nonzero(a, axis=1))
    return n >= k >= z


def check_tuple(g: DWGraph, g_index: List[int], g0: DWGraph, g0_index: List[int],
                edges_matching, t: tuple, start_i: int):
    q0 = g0_index[start_i]
    v0 = g_index[t[0]]

    g_lab = g.get_labels()
    g0_lab = g0.get_labels()

    for k in range(1, len(t)):
        q1 = g0_index[start_i + k]
        v1 = g_index[t[k]]
        if q1 in g0[q0]:
            if v1 not in g[v0]:
                return False
            if edges_matching is None:
                continue
            for eq_dict in g0[q0][q1].values():
                wq = eq_dict['weight']
                matched = False
                for eg_dict in g[v0][v1].values():
                    wg = eg_dict['weight']
                    if edges_matching((g_lab[v0], g_lab[v1], wg), (g0_lab[q0], g0_lab[q1], wq)):
                        matched = True
                        break

                if not matched:
                    return False

    return True


def unique_tuples_mtx(g: DWGraph, g_index: List[int], g0: DWGraph, g0_index,
                      edges_matching, u: np.ndarray, i: int = 0) -> Iterable[Tuple[int]]:
    if i == u.shape[0]:
        yield ()
        return

    idx = np.where(u[i, :])[0].tolist()
    u[i, :] = 0

    for j in idx:
        u0 = u.copy()
        u0[:, j] = 0
        if check_mtx(u0, u.shape[0] - i - 1):
            for r in unique_tuples_mtx(g, g_index, g0, g0_index, edges_matching, u0, i+1):
                temp_result = (j, ) + r
                if check_tuple(g, g_index, g0, g0_index, edges_matching, temp_result, i):
                    yield temp_result
        del u0
