import networkx as nx
import pandas as pd
import numpy as np
from typing import Iterable, List, Callable, Tuple, Dict, Union, Set
from copy import deepcopy
from collections import defaultdict
from bisect import bisect_right, bisect_left


class DWGraph(nx.MultiDiGraph):
    """
    This class extends networkx.MultiDiGraph to perform reading/writing from file
    and other useful algorithms that not implemented in networkX library.
    """

    EdgesMatchingType = Callable[[Tuple[int, int, float], Tuple[int, int, float]], bool]

    class DFSPerformerOriginal:
        """
        It's a modification of DFS algorithm used here to find all the templates in background graph.
        """

        def __init__(self, g: 'DWGraph', v0: int, g0: 'DWGraph', q0: int, v_to_v0_mapping: Dict[int, int]):
            self.g = g
            self.g0 = g0
            self.v0 = v0
            self.q0 = q0
            self.used_g0 = {}
            self.template_size = len(g0.nodes)
            self.f = v_to_v0_mapping
            for node in g0.nodes:
                self.used_g0[node] = False

            self.used_g = []

            first_used = {}
            for node in g.nodes:
                first_used[node] = None

            self.dfs(q0, first_used, v0, 1)

        def dfs(self, q0, used_g, v0, n):
            self.used_g0[q0] = True
            used_g[v0] = q0

            # When we achieve the appropriate number of vertices,
            # we append vertices list to result.
            if n == self.template_size:
                self.used_g.append(used_g)
                return  # check here?
                
            # For q in adj(q0)
            for q in self.g0[q0].keys():
                if self.used_g0[q]:
                    continue

                # For v in adj(v0)
                for v in self.g[v0].keys():
                    if used_g[v] is not None or self.f.get(v, q-1) != q:
                        continue
                    self.dfs(q, used_g.copy(), v, n+1)

    class DFSPerformer:
        """
        It's a modification of DFS algorithm used here to find all the templates in background graph.
        """

        def __init__(self, g: 'DWGraph', v0: int, g0: 'DWGraph', q0: int, v_to_v0_mapping: Dict[int, set], output_file):
            self.g = g
            self.g0 = g0
            self.v0 = v0
            self.q0 = q0
            self.output_file = output_file

            self.template_size = len(g0.nodes)
            self.f = v_to_v0_mapping

            self.used_g = []

            first_used = {}
            for node in g.nodes:
                first_used[node] = None

            self.dfs(-1, q0, first_used, v0, 1)

        def dfs(self, q_prev, q0, used_g, v0, n):
            used_g[v0] = q0

            # When we achieve the appropriate number of vertices,
            # we append vertices list to result.
            if n == self.template_size:
                self.used_g.append(used_g)
                print('({}) '.format(len(self.used_g)), end='')
                r = DWGraph.get_mapping_tuples_by_vertices_dict(self.used_g[-1], self.g, self.g0)
                if r is not None:
                    print(r)
                    # Maybe print to file?
                    if self.output_file is not None:
                        with open(self.output_file, 'a') as f:
                            f.write(str(r)+'\n')
                else:
                    print('')

                return  # check here?

            # For q in adj(q0)
            for q in self.g0[q0].keys():
                if q == q_prev:
                    continue
                # For v in adj(v0)
                for v in self.g[v0].keys():
                    if used_g[v] is not None or q not in self.f.get(v, set()):
                        continue
                    self.dfs(q0, q, used_g.copy(), v, n+1)
                    print('DFS finished with n = {} of total = {}'.format(n, self.template_size))

    def __init__(self):
        super().__init__()
        self.is_weighted_prop = False
        self.is_directed_prop = None
        self._labels = {}
        self.info_memoization = {}

    def is_weighted(self):
        return self.is_weighted_prop

    def is_directed(self):
        return self.is_directed_prop

    def fresh_copy(self):
        """
        Return a fresh copy graph with the same data structure.

        A fresh copy has no nodes, edges or graph attributes. It is
        the same data structure as the current graph. This method is
        typically used to create an empty version of the graph.
        """
        return DWGraph()

    @staticmethod
    def read_graph_edges(file_path: str,
                         weighted=True,
                         skip_first_line=True,
                         col_types=(int, int, float)):
        """
        Returns generator reading edges from file line-by-line
        :param file_path: Path to edges list file
        :param weighted: Are edge weights specified?
        :param skip_first_line: Is first line a header?
        :param col_types: Types of identifiers and weights columns.
        :return: Generator
        """

        n_entries = 3 if weighted else 2

        with open(file_path, "r") as f:
            if skip_first_line:
                f.readline()
            while True:
                line = f.readline()
                if line == '':
                    break
                entries = line.split()
                if len(entries) != n_entries:
                    continue
                res = []
                for i, entry in enumerate(entries):
                    res.append(col_types[i](entry))
                yield tuple(res)

    @staticmethod
    def from_file_edge_list(file_path, is_weighted=True, skip_first_line=True, make_undirected=False):
        """
        Initialize graph from the list of its (un)weighted edges.
        Each line of input file should has the format
        vertex1 vertex2 (edge_weight)
        """

        file_reader = DWGraph.read_graph_edges(file_path, is_weighted, skip_first_line)

        w_list = []
        for e in file_reader:
            w_list.append(e)
            if make_undirected:
                if is_weighted:
                    e = (e[1], e[0], e[2])
                else:
                    e = (e[1], e[0])
                w_list.append(e)

        g = DWGraph()
        if is_weighted:
            g.add_weighted_edges_from(w_list)
        else:
            g.add_edges_from(w_list)
        g.is_weighted_prop = is_weighted
        g.is_directed_prop = not make_undirected
        return g

    @staticmethod
    def lcc_original(g: 'DWGraph', g0: 'DWGraph', t: dict, k_max: int,
                     edges_matching: EdgesMatchingType = None) -> dict:
        """
        Performs Local constraint checking algorithm
        :param g: background graph
        :param g0: template graph
        :param t: the subset of vertices of g
        :param k_max: maximal number of iterations
        :param edges_matching: function for matching edges instead of matching vertices
        :return: new T (subset of vertices of g)
        """

        # Initialization
        g_labels = g.get_labels()
        g0_labels = g0.get_labels()
        template_label_to_vertex = g0.get_rev_labels()
        f = {}

        # Initialize default value of f(v)
        _default = np.min(list(template_label_to_vertex.values())) - 1

        # We also initialize set T as a list to iterate
        # faster over persisting vertices
        t_list = []
        for v in t.keys():
            if t[v] == 1:
                t_list.append(v)
                f[v] = template_label_to_vertex.get(g_labels[v], _default)

        # The main loop
        for k in range(k_max):
            d_t = 0
            f_new = f.copy()
            for v in t_list:
                if t[v] == 0 or f[v] == _default:
                    continue
                for q in g0[f[v]].keys():
                    flag = False
                    for v1 in g[v].keys():
                        if edges_matching is None:
                            flag = f.get(v1, _default) == q
                        else:
                            g_e = (g_labels[v], g_labels[v1], g.get_max_weight(v, v1))
                            g0_e = (g0_labels[f[v]], g0_labels[q], g0.get_max_weight(f[v], q))
                            flag = edges_matching(g_e, g0_e)
                        if flag:
                            break

                    if not flag:
                        f_new[v] = _default
                        t[v] = 0
                        d_t -= 1
                        break
            if d_t == 0:
                break
            f = f_new

        # Refine new subset
        new_t = {}
        for k, v in t.items():
            if v == 1:
                new_t[k] = 1
        return new_t

    @staticmethod
    def lcc(g: 'DWGraph', g0: 'DWGraph', t: Dict[int, int], f: Dict[int, set], d: Dict[int, int], k_max: int,
            edges_matching: EdgesMatchingType = None) \
            -> Tuple[Dict[int, int], Dict[int, set], Dict[int, int], bool, bool]:
        """
        Performs Local constraint checking algorithm
        :param g: background graph
        :param g0: template graph
        :param t: the subset of vertices of g
        :param f: specific f-function in extended V -> 2^V0 format
        :param d: break-out dictionary. Counts number of candidates in background graph.
        :param k_max: maximal number of iterations
        :param edges_matching: function for matching edges instead of matching vertices
        :return: new T (subset of vertices of g) and new f-function
        """

        # Initialization
        g_labels = g.get_labels()
        g0_labels = g0.get_labels()
        exit_flag = False
        is_effective = False

        # We initialize set T as a list to iterate
        # faster over persisting vertices
        t_list = []
        for v in t.keys():
            if t[v] == 1:
                t_list.append(v)
            else:
                f[v] = set()

        # The main loop
        for k in range(k_max):
            d_t = 0

            # Deep copy for copying sets
            f_new = deepcopy(f)
            it = 0
            for v in t_list:
                v_info = g.get_vertex_label_info(v, f)

                it += 1
                if it % 1000 == 0:
                    print(str(it) + " vertices proceeded")

                if v not in t or t[v] == 0 or len(f[v]) == 0:
                    continue
                for q0 in f[v]:
                    q0_info = g0.get_vertex_label_info(q0)
                    is_greater = DWGraph.vertex_greater_by_dict(v_info, q0_info)

                    for q in g0[q0].keys():
                        flag = False
                        if is_greater:
                            for v1 in g[v].keys():
                                if v1 not in t or t[v1] != 1 or q not in f.get(v1, set()):
                                    continue
                                if edges_matching is None:
                                    flag = True
                                else:
                                    flag = False
                                    g0_e = (g0_labels[q0], g0_labels[q], g0.get_max_weight(q0, q))
                                    for gw in g.get_all_weights(v, v1):
                                        g_e = (g_labels[v], g_labels[v1], gw)
                                        if edges_matching(g_e, g0_e):
                                            flag = True
                                            break
                                if flag:
                                    break

                        if not flag:
                            f_new[v].remove(q0)
                            d[q0] -= 1
                            if d[q0] == 0:
                                exit_flag = True
                            if len(f_new[v]) == 0:
                                t[v] = 0
                            d_t -= 1
                            is_effective = True
                            break
                    if exit_flag:
                        break

                if exit_flag:
                    break

            if d_t == 0 or exit_flag:
                break
            f = f_new

        # Refine new subset
        new_t = {}
        for k, v in t.items():
            if v == 1:
                new_t[k] = 1
        return new_t, f, d, exit_flag, is_effective

    @staticmethod
    def lcc_r(g: 'DWGraph', g0: 'DWGraph', fr: Dict[int, Set[int]], k_max: int,
              edges_matching: EdgesMatchingType = None) \
            -> Tuple[Dict[int, Set[int]], bool, bool]:
        """
        Performs Local constraint checking algorithm
        :param g: background graph
        :param g0: template graph
        :param fr: specific fr-function in extended Q -> 2^V format
        :param k_max: maximal number of iterations
        :param edges_matching: function for matching edges instead of matching vertices
        :return: new fr-function, exit flag and effectiveness
        """

        # Initialization
        g_labels = g.get_labels()
        g0_labels = g0.get_labels()
        exit_flag = False
        is_effective = False

        # The main loop
        for k in range(k_max):
            d_t = 0

            for q0 in g0:
                for q in g0[q0]:
                    matched = False
                    for v0 in list(fr[q0]):
                        flag = False
                        for v in g[v0]:
                            if v not in fr[q]:
                                continue
                            if edges_matching is None:
                                flag = True
                            else:
                                g0_e = (g0_labels[q0], g0_labels[q], g0.get_max_weight(q0, q))
                                for gw in g.get_all_weights(v0, v):
                                    g_e = (g_labels[v0], g_labels[v], gw)
                                    if edges_matching(g_e, g0_e):
                                        flag = True
                                        break
                            if flag:
                                break

                        if not flag:
                            fr[q0].remove(v0)
                            d_t += 1
                            is_effective = True
                            if not fr[q0]:
                                exit_flag = True
                                break
                        else:
                            matched = True

                    exit_flag = exit_flag or not matched
                    if exit_flag:
                        break

                if exit_flag:
                    break

            if d_t == 0 or exit_flag:
                break

        return fr, exit_flag, is_effective

    @staticmethod
    def cc(g: 'DWGraph', g0: 'DWGraph',
           t: Dict[int, int], f: Dict[int, set], d: Dict[int, int], k0: Iterable[List[int]],
           edges_matching: EdgesMatchingType = None) \
            -> Tuple[Dict[int, int], Dict[int, set], Dict[int, int], bool, bool]:
        """
        Performs cycle checking algorithm
        :param g: background graph
        :param g0: template graph
        :param t: the subset of vertices of g
        :param f: function f from LCC algorithm
        :param d: break-out dictionary. Counts number of candidates in background graph.
        :param k0: cycles templates
        :param edges_matching: function for matching edges instead of matching vertices
        :return: new T (subset of vertices of g)
        """

        exit_flag = False
        is_effective = False

        # iterate through all cycles templates
        it = 0
        for c0 in k0:
            print(str(it) + " cycles proceeded")
            it += 1
            # initialize cycles
            # labels of background graph
            g_labels = g.get_labels()
            # labels of template graph
            g0_labels = g0.get_labels()
            # convert t-dictionary to list
            t_list = [k for k in t.keys() if t[k] == 1]

            # get the first edge of the template cycle
            if len(c0) == 1:
                continue
            q0, q1 = c0[0:2]

            # initialization itself
            it2 = 0
            for v0 in t_list:
                if it2 % 500 == 0:
                    print(str(it2) + " vertices proceeded")
                    print(str(d[q0]) + ' candidates for vertex ' + str(q0))
                it2 += 1
                # set of chain ends
                if q0 not in f.get(v0, set()):
                    continue

                a = [v0]

                # main loop
                for s in range(len(c0)):
                    # print(len(c0))
                    # qb is the s-th edge's start
                    qb = c0[s % len(c0)]
                    qb_label = g0_labels[qb]

                    # qe is the s-th edge's end
                    qe = c0[(s + 1) % len(c0)]
                    qe_label = g0_labels[qe]

                    # get weight of (q0, q1) edge
                    w0 = None
                    if edges_matching is not None:
                        w0 = g0.get_max_weight(qb, qe)

                    b = []
                    # print(len(a))
                    for v in a:
                        for v1 in g[v].keys():
                            if qe not in f.get(v1, set()):
                                continue
                            if s == len(c0) - 1 and v0 != v1:
                                continue
                            if t.get(v1, 0) == 1:
                                if edges_matching is None:
                                    condition = g_labels[v1] == qe_label
                                else:
                                    condition = False
                                    g0_e = (qb_label, qe_label, w0)
                                    for gw in g.get_all_weights(v, v1):
                                        g_e = (g_labels[v], g_labels[v1], gw)
                                        if edges_matching(g_e, g0_e):
                                            condition = True
                                            break
                                if condition:
                                    b.append(v1)
                    a.clear()
                    a = b.copy()

                if v0 not in a:
                    is_effective = True
                    f[v0].remove(q0)
                    t[v0] = 0 if len(f[v0]) == 0 else 1
                    d[q0] -= 1
                    if d[q0] == 0:
                        # print(q0)
                        exit_flag = True
                if exit_flag:
                    break

            if exit_flag:
                break

        # Refine new subset
        new_t = {}
        for k, v in t.items():
            if v == 1:
                new_t[k] = 1
        return new_t, f, d, exit_flag, is_effective

    @staticmethod
    def cc_r(g: 'DWGraph', g0: 'DWGraph', fr: Dict[int, Set[int]], k0: Iterable[List[int]],
             edges_matching: EdgesMatchingType = None) \
            -> Tuple[Dict[int, Set[int]], bool, bool]:
        """
        Performs cycle checking algorithm
        :param g: background graph
        :param g0: template graph
        :param fr: function f from LCC algorithm
        :param k0: cycles templates
        :param edges_matching: function for matching edges instead of matching vertices
        :return: new T (subset of vertices of g)
        """

        exit_flag = False
        is_effective = False

        # iterate through all cycles templates
        it = 0
        for c0 in k0:
            print(str(it) + " cycles proceeded")
            it += 1
            # initialize cycles
            # labels of background graph
            g_labels = g.get_labels()
            # labels of template graph
            g0_labels = g0.get_labels()

            # get the first edge of the template cycle
            if len(c0) == 1:
                continue
            q0, q1 = c0[0:2]

            for v0 in list(fr[q0]):
                a = [v0]
                for s in range(len(c0)):
                    # print(len(c0))
                    # qb is the s-th edge's start
                    qb = c0[s % len(c0)]
                    qb_label = g0_labels[qb]

                    # qe is the s-th edge's end
                    qe = c0[(s + 1) % len(c0)]
                    qe_label = g0_labels[qe]

                    # get weight of (q0, q1) edge
                    w0 = None
                    if edges_matching is not None:
                        w0 = g0.get_max_weight(qb, qe)

                    b = []
                    # print(len(a))
                    for v in a:
                        for v1 in fr[qe]:
                            if v1 not in g[v]:
                                continue
                            if s == len(c0) - 1 and v0 != v1:
                                continue
                            if edges_matching is None:
                                condition = g_labels[v1] == qe_label
                            else:
                                condition = False
                                g0_e = (qb_label, qe_label, w0)
                                for gw in g.get_all_weights(v, v1):
                                    g_e = (g_labels[v], g_labels[v1], gw)
                                    if edges_matching(g_e, g0_e):
                                        condition = True
                                        break
                            if condition:
                                b.append(v1)
                    a.clear()
                    a = b.copy()

                if v0 not in a:
                    is_effective = True
                    fr[q0].remove(v0)
                    if not fr[q0]:
                        # print(q0)
                        exit_flag = True
                if exit_flag:
                    break

            if exit_flag:
                break

        return fr, exit_flag, is_effective

    @staticmethod
    def vertex_elimination(g: 'DWGraph', g0: 'DWGraph', k_max: int = 2,
                           edges_matching: EdgesMatchingType = None) -> Union[Dict[int, set], None]:
        """
        This method applies LCC and CC methods until the vertices are being removed
        :param g: background graph
        :param g0: template graph
        :param k_max: number of iterations for LCC
        :param edges_matching: function for matching edges instead of matching vertices
        :return: new T (subset of vertices of g)
        """
        t = g.get_full_subset()
        print('Full subset got')
        k0 = g0.get_cycles()
        print('Cycles got')

        g_labels = g.get_labels()
        template_label_to_vertex = g0.get_rev_labels_extended()
        f = {}
        d = defaultdict(lambda: 0)
        for v in t.keys():
            r = template_label_to_vertex.get(g_labels[v], set()).copy()
            f[v] = set()

            for p in r:
                # if g.degree(v) >= g0.degree(p):
                if DWGraph.vertex_greater(g, v, g0, p):
                    f[v].add(p)
                    d[p] += 1

            if len(f[v]) == 0:
                t[v] = 0

        if len(d.keys()) != len(g0.nodes):
            return None

        while True:
            print('LCC started')
            t, f, d, exit_flag, is_lcc_effective = DWGraph.lcc(g, g0, t, f, d, k_max, edges_matching)
            # print('LCC done', t, f)
            print('LCC done')
            if exit_flag:
                return None
            print('CC started')
            t, f, d, exit_flag, is_cc_effective = DWGraph.cc(g, g0, t, f, d, k0, edges_matching)
            # print('CC done', t)
            print('CC done')
            if exit_flag:
                return None

            # f, d, t, exit_flag, is_refine_effective = DWGraph.refine_candidate_function(f, t, d)
            if exit_flag:
                return None
            if not is_lcc_effective and not is_cc_effective:  # and not is_refine_effective:
                break

        # reorganize output format
        # We'll return dictionary T -> 2^V0

        # set T as list:
        t_list = sorted(list(t.keys()))

        # labels and reverse labels for getting result
        result = {}
        for v in t_list:
            result[v] = f[v]

        return result

    @staticmethod
    def vertex_elimination_r(g: 'DWGraph', g0: 'DWGraph', k_max: int = 2,
                             edges_matching: EdgesMatchingType = None) -> Union[Dict[int, set], None]:
        """
        This method applies LCC and CC methods until the vertices are being removed
        :param g: background graph
        :param g0: template graph
        :param k_max: number of iterations for LCC
        :param edges_matching: function for matching edges instead of matching vertices
        :return: new T (subset of vertices of g)
        """
        k0 = g0.get_cycles()
        print('Cycles got')

        g0_labels = g0.get_labels()
        bg_label_to_vertex = g.get_rev_labels_extended()
        fr = {}
        for q in g0.nodes:
            r = bg_label_to_vertex.get(g0_labels[q], set()).copy()
            fr[q] = set()

            for v in r:
                # if g.degree(v) >= g0.degree(p):
                if DWGraph.vertex_greater(g, v, g0, q):
                    fr[q].add(v)

            if not fr[q]:
                return None

        while True:
            print('LCC started')
            fr, exit_flag, is_lcc_effective = DWGraph.lcc_r(g, g0, fr, k_max, edges_matching)
            # print('LCC done', t, f)
            print('LCC done')
            if exit_flag:
                return None
            print('CC started')
            fr, exit_flag, is_cc_effective = DWGraph.cc_r(g, g0, fr, k0, edges_matching)
            # print('CC done', t)
            print('CC done')
            if exit_flag:
                return None

            # f, d, t, exit_flag, is_refine_effective = DWGraph.refine_candidate_function(f, t, d)
            if exit_flag:
                return None
            if not is_lcc_effective and not is_cc_effective:  # and not is_refine_effective:
                break

        # reorganize output format
        # We'll return dictionary T -> 2^V0
        result = {}
        for q, v_s in fr.items():
            for v in v_s:
                result.setdefault(v, set()).add(q)

        return result

    @staticmethod
    def get_vertices_list_original(g: 'DWGraph', g0: 'DWGraph', t: Dict[int, int]) -> List[List[Tuple[int, int]]]:
        """
        Returns the result of vertex elimination as a list of lists of vertices of subgraphs of background graph
        :param g: Background graph
        :param g0: Template graph
        :param t: Subset of vertices of background mapped into V0
        :return: List of list of tuples. Each tuple is a pair (q, v) where q is an index in G0 and v is an index in G.
                 List is sorted by q (in ascending order)
        """

        # Reverse the mapping. It's not bijective, that's why we map each element of V0 to a set of V
        rev_map = {}
        for k, v in t.items():
            rev_map.setdefault(v, set()).add(k)

        # Initialize the result with empty list
        res = []

        # Get t(V)
        v0 = iter(rev_map.items())

        try:
            # Get the first vertex from V0
            q0, q0_set = next(v0)

            for v0 in q0_set:
                # Start DFS in (G, v0) and (G0, q0) in the same time
                dfs_performer = DWGraph.DFSPerformerOriginal(g, v0, g0, q0, t)

                for vertices_dict in dfs_performer.used_g:
                    reverse_dict = {}
                    for k, v in vertices_dict.items():
                        reverse_dict[v] = k

                    vertices_list = [(v, k) for k, v in vertices_dict.items() if v is not None]
                    vertices_list.sort(key=lambda pair: pair[0])
                    # Check if all the edges from G0 exist in this subset of G
                    passed = True
                    for q, v in vertices_list:
                        for q1 in g0[q].keys():
                            passed_edge = False
                            v1 = reverse_dict[q1]

                            for v11 in g[v].keys():
                                if v1 == v11:
                                    passed_edge = True
                                    break
                            if not passed_edge:
                                passed = False
                                break

                        if not passed:
                            break

                    # If check is done, add vertices list to result
                    if passed:
                        res.append(vertices_list)

            return res

        except StopIteration:
            return res

    @staticmethod
    def get_mapping_tuples_by_vertices_dict(vertices_dict: Dict[int, int], g, g0):
        reverse_dict = {}
        for k, v in vertices_dict.items():
            reverse_dict[v] = k

        vertices_list = [(v, k) for k, v in vertices_dict.items() if v is not None]
        vertices_list.sort(key=lambda pair: pair[0])
        # Check if all the edges from G0 exist in this subset of G
        passed = True
        for q, v in vertices_list:
            for q1 in g0[q].keys():
                passed_edge = False
                v1 = reverse_dict[q1]

                for v11 in g[v].keys():
                    if v1 == v11:
                        passed_edge = True
                        break
                if not passed_edge:
                    passed = False
                    break

            if not passed:
                break

        # If check is done, add vertices list to result
        if passed:
            return vertices_list

    @staticmethod
    def get_vertices_list(g: 'DWGraph', g0: 'DWGraph', t: Dict[int, set],
                          output_file: str = None, edges_matching: EdgesMatchingType = None) -> \
            List[List[Tuple[int, int]]]:
        """
        Returns the result of vertex elimination as a list of lists of vertices of subgraphs of background graph
        :param g: Background graph
        :param g0: Template graph
        :param t: Subset of vertices of background multi-mapped into V0
        :param output_file: File for vertices output
        :param edges_matching: Edges matching function for this configuration
        :return: List of list of tuples. Each tuple is a pair (q, v) where q is an index in G0 and v is an index in G.
                 List is sorted by q (in ascending order)
        """
        from unique_tuples import unique_tuples

        with open(output_file, 'w'):
            pass

        # Reverse the mapping. It's not bijective, that's why we map each element of V0 to a set of V
        rev_map = {}
        g_candidates = []

        for k, v_set in t.items():
            if v_set:
                g_candidates.append(k)
            for v in v_set:
                rev_map.setdefault(v, set()).add(k)
        g0_candidates = list(rev_map.keys())

        # Initialize the result with empty list
        res = []

        # Candidate subgraph
        g_subgraph = nx.subgraph(g, g_candidates)
        g_subgraph.get_labels = g.get_labels

        u_tuples_sets = [rev_map[k] for k in g0_candidates]
        for tpl in unique_tuples(g_subgraph, g0_candidates, g0, edges_matching, u_tuples_sets):
            res_list = [(g0_candidates[i], v) for i, v in enumerate(tpl)]
            if output_file is not None:
                with open(output_file, 'a') as f:
                    f.write(str(res_list) + '\n')
            res.append(res_list)

        return res

    @staticmethod
    def get_vertices_list_recursive(g: 'DWGraph', g0: 'DWGraph', t: Dict[int, set],
                                    output_file: str = None, edges_matching: EdgesMatchingType = None,
                                    comp_approx: int = 10) -> \
            List[List[Tuple[int, int]]]:
        """
        Returns the result of vertex elimination as a list of lists of vertices of subgraphs of background graph
        :param g: Background graph
        :param g0: Template graph
        :param t: Subset of vertices of background multi-mapped into V0
        :param output_file: File for vertices output
        :param edges_matching: Edges matching function for this configuration
        :param comp_approx: Approximate number of vertices in each component
        :return: List of list of tuples. Each tuple is a pair (q, v) where q is an index in G0 and v is an index in G.
                 List is sorted by q (in ascending order)
        """
        from recursive_unique_tuples import recursive_unique_tuples

        with open(output_file, 'w'):
            pass

        # Reverse the mapping. It's not bijective, that's why we map each element of V0 to a set of V
        rev_map = {}
        g_candidates = []

        for k, v_set in t.items():
            if v_set:
                g_candidates.append(k)
            for v in v_set:
                rev_map.setdefault(v, set()).add(k)
        g0_candidates = list(rev_map.keys())

        # Initialize the result with empty list
        res = []

        # Candidate subgraph
        g_subgraph = nx.subgraph(g, g_candidates)
        g_subgraph.get_labels = g.get_labels

        # for tpl in unique_tuples(g_subgraph, g0_candidates, g0, edges_matching, u_tuples_sets):

        for tpl in recursive_unique_tuples(g_subgraph, g0, rev_map, g0_candidates, edges_matching, comp_approx):
            res_list = [(g0_candidates[i], v) for i, v in enumerate(tpl)]
            if output_file is not None:
                with open(output_file, 'a') as f:
                    f.write(str(res_list) + '\n')
                    res.append(res_list)

        return res

    @staticmethod
    def get_graphs_by_vertices_list(g0: 'DWGraph', vertices: List[List[Tuple[int, int]]]) \
            -> List['DWGraph']:
        """
        Transforms vertices lists to graphs
        :param g0: Template graph
        :param vertices: List which structure is described in get_vertices_list method description
        :return: List of subgraphs of G
        """

        res = []
        for v_list in vertices:
            mapping = dict(v_list)
            res.append(nx.relabel_nodes(g0, mapping, True))
        return res

    def get_max_weight(self, u, v):
        return max(self.get_edge_data(u, v).values(), key=lambda e: e['weight'])['weight']

    def get_all_weights(self, u, v):
        return [e['weight'] for e in self.get_edge_data(u, v).values()]

    @staticmethod
    def vertex_greater_by_dict(x_info: Dict[int, int], y_info: Dict[int, int]) -> bool:
        for k, n in y_info.items():
            if k not in x_info:
                return False
            if x_info[k] < y_info[k]:
                return False

        return True

    @staticmethod
    def refine_candidate_function(f: Dict[int, Set[int]], t: Dict[int, int], d: Dict[int, int]) -> \
            Tuple[Dict[int, Set[int]], Dict[int, int], Dict[int, int], bool, bool]:
        continue_flag = True
        exit_flag = False
        cnt = -1
        while continue_flag:
            cnt += 1
            print('Refine, iteration #{}'.format(cnt + 1))
            continue_flag = False
            new_f = deepcopy(f)
            for v, v_candidates in f.items():
                if len(v_candidates) == 1:
                    q0 = next(iter(v_candidates))
                    for v1, v1_candidates in f.items():
                        if v != v1 and q0 in v1_candidates:
                            continue_flag = True
                            new_f[v1].remove(q0)
                            d[q0] -= 1
                            if d[q0] == 0:
                                exit_flag = True
                            if len(new_f[v1]) == 0:
                                t[v1] = 0
            f = new_f

        return f, d, t, exit_flag, cnt != 0

    @staticmethod
    def vertex_greater(g1: 'DWGraph', x: int, g2: 'DWGraph', y: int) -> bool:
        x_info = g1.get_vertex_label_info(x)
        y_info = g2.get_vertex_label_info(y)
        return DWGraph.vertex_greater_by_dict(x_info, y_info)

    def get_vertex_label_info(self, v: int, f: Dict[int, Set[int]] = None, memo: bool=True) -> Dict[int, int]:
        memo = memo and f is None
        if memo and self.info_memoization.get(v, False):
            return self.info_memoization[v]

        v_info = {}
        labels = self.get_labels()
        for u, d in self.adj[v].items():
            if f is not None:
                u_candidates = f.get(u, False)
                if not u_candidates:
                    continue

            prev_count = v_info.get(labels[u], 0)
            v_info[labels[u]] = prev_count + len(d)

        if memo:
            self.info_memoization[v] = v_info

        return v_info

    def write_to_file(self, file_path):
        """
        Write to file the list of graph's [un]weighted edges.
        Each line of output file will have the format
        vertex1 vertex2 [edge_weight]
        """
        edges = self.edges
        adj = self.adj
        with open(file_path, 'w') as f:
            for e in edges:
                x, y, i = e
                if self.is_weighted():
                    w = adj[x][y][i]['weight']
                    f.write("{} {} {}\n".format(x, y, w))
                else:
                    f.write("{} {}\n".format(x, y))

    def get_labels(self) -> dict:
        """
        Returns V -> Label dictionary
        """
        return self._labels

    def get_rev_labels(self) -> dict:
        """
        Returns Label -> V dictionary
        Throws NonUniqueLabelsError in case if several vertices
        have the same label.
        """
        rev = {}
        for v, label in self.get_labels().items():
            if label in rev:
                raise NonUniqueLabelsError()
            rev[label] = v
        return rev

    def get_rev_labels_extended(self):
        """
        Returns Label -> 2^V dictionary
        For each label, result[l] is a set of vertices with label l.
        """
        rev = {}
        for v, label in self.get_labels().items():
            rev.setdefault(label, set()).add(v)
        return rev

    def get_full_subset(self):
        """
        Returns initial subset for this graph
        S[v] = 1 if v is vertex and -1 of not.
        """
        nodes = self.nodes
        subs = {}

        for x in nodes:
            subs[x] = 1

        return subs

    def add_default_labels(self) -> None:
        """
        Adds default label value V to each vertex V.
        """
        self.add_labels_by_identifiers(lambda v: v)

    def add_labels_by_identifiers(self, func: Callable[[int], int]):
        """
        Adds label value func(V) to each vertex V.
        """
        for v in self.nodes:
            self._labels[v] = func(v)

    def add_labels_from_file(self, file_path):
        """
        Reads the file each line of which has the format:
        Vertex Label
        Assigns Labels to corresponding vertices
        """
        df = pd.read_csv(filepath_or_buffer=file_path, delim_whitespace=True,
                         names=['v', 'label', 'w'], dtype={'v': np.int32, 'label': np.int32}, header=None)
        vl, lab_i = df['v'], iter(df['label'])
        for v in vl:
            self._labels[v] = next(lab_i)

    def has_cycle(self, cycle: List[int]) -> bool:
        """
        Does this graph has such cycle?
        :param cycle: Cycle as a list of vertices
        :return: True if this graph has such cycle
        """
        for i in range(len(cycle)):
            fr = cycle[i]
            to = cycle[(i+1) % len(cycle)]
            if to not in self[fr]:
                return False

        return True

    def get_cycles(self) -> Iterable[List[int]]:
        """
        Returns the generator of cycles of this graph used by CC algorithm.
        :return: Generator of lists-cycles.
        """

        simple_edges = [e[0:2] for e in self.edges]
        cycles = nx.cycle_basis(nx.Graph(simple_edges))
        if self.is_directed():
            # cycles = nx.simple_cycles(nx.DiGraph(simple_edges))
            res = []
            for c in cycles:
                if self.has_cycle(c):
                    res.append(c)
            return res
        else:
            return cycles

    @staticmethod
    def get_ratios(g: 'DWGraph', g0: 'DWGraph',
                   edge_type_g, e_type_g_labels: bool,
                   edge_type_g0, e_type_g0_labels: bool, w_eps: float) -> \
            Union[Dict[Tuple[int, int], Dict[float, float]], Dict[Tuple[int, int], float]]:
        """
        Returns the matching ratios of edges in g0 to edges of g.
        :param g: Background graph
        :param g0: Template graph
        :param edge_type_g: (u, v) -> type mapping for G
        :param e_type_g_labels: edge_type_g takes pair of labels, not vertices
        :param edge_type_g0: (u, v) -> type mapping for G0
        :param e_type_g0_labels: edge_type_g0 takes pair of labels, not vertices
        :param w_eps: epsilon for matching weights
        :return: Dictionary (u, v) |-> {w_1 -> r_1, w_2 -> r_2, ..., w_s -> r_s} OR
                            (u, v) |-> r [in case of unweighted graph]
        """
        g_labels = g.get_labels()
        g0_labels = g0.get_labels()

        '''
        When we match edges, we base on the following:
        |w - w0| < eps  <=>
        w0 - eps < w < w0 + eps  
        '''

        # e2 = 1/(w_eps*2)
        edge_type_check = True

        if edge_type_g is None or edge_type_g0 is None:
            edge_type_check = False

            # This function is not used, just for conformity
            def e_type_g(_u, _v):
                return 0 * (_u + _v)
            e_type_g0 = e_type_g
        else:
            e_type_g = (lambda _u, _v: edge_type_g(g_labels[_u], g_labels[_v])) \
                if e_type_g_labels else edge_type_g
            e_type_g0 = (lambda _u, _v: edge_type_g0(g0_labels[_u], g0_labels[_v])) \
                if e_type_g0_labels else edge_type_g0

        print("Making stats by graph G...")
        g_stats = {}
        total = 0
        for u, v, w in g.edges.data('weight', default=None):
            k = (g_labels[u], g_labels[v])

            if edge_type_check:
                c = e_type_g(u, v)
                k += (c, )

            weights_dict = g_stats.setdefault(k, {})

            weights_dict[w] = weights_dict.get(w, 0) + 1
            total += 1

        print("Converting stats to searchable form...")
        g_list_stats = {}
        for k, weights_dict in g_stats.items():
            items_list = list(weights_dict.items())
            if items_list[0][0] is not None:
                items_list.append((-1e10, 0))
                items_list.sort(key=lambda t: t[0])
                w_list = [w for w, _ in items_list]
                prefix_sums = [count for _, count in items_list]
                for i in range(1, len(prefix_sums)):
                    prefix_sums[i] += prefix_sums[i - 1]
                g_list_stats[k] = (w_list, prefix_sums)
            else:
                g_list_stats[k] = items_list[0][1]

        del g_stats

        print("Making resulting dictionary...")
        res = {}
        for u, v, w in g0.edges.data('weight', default=None):
            k = (g0_labels[u], g0_labels[v])

            if edge_type_check:
                c = e_type_g0(u, v)
                k += (c, )

            w_search = g_list_stats.get(k, None)
            if w_search is None:
                count = 0
            elif isinstance(w_search, int):
                count = w_search
            else:
                w_list, prefix_sums = w_search
                i_from = bisect_right(w_list, w - w_eps)
                i_to = bisect_left(w_list, w + w_eps)
                count = prefix_sums[i_to - 1] - prefix_sums[i_from - 1]

            r = count / total

            if w is not None:
                res.setdefault((u, v), {}).__setitem__(w, r)
            else:
                res[(u, v)] = r

        return res


class NonUniqueLabelsError(Exception):
    """
    This exception is thrown if you try to invert labels
    of non-unique-labeled graph.
    """

    def __init__(self):
        pass
