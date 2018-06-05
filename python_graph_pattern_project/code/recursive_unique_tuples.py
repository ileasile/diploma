from typing import Dict, List, Tuple, Set, Union
from DirectedWeightedGraph import DWGraph
import networkx as nx


class BFS:
    def __init__(self, g: List[List[Tuple[int, List[Tuple[int, int, float]]]]], lim: int):
        """
        Runs the BFS for all vertices in graph G and collects BFS info
        :param g: Graph in form af adjacency lists
        :param lim: Maximal number of vertices in one component
        """
        self.used = [-1 for _ in range(len(g))]
        self.lim = lim
        self.g = g
        comp_n = 0
        for v0, _ in enumerate(g):
            if self.used[v0] == -1:
                self.run(v0, comp_n)
                comp_n += 1

        self.sz = comp_n

    def run(self, v0: int, comp_n: int) -> None:
        """
        Runs BFS from vertex v0, labeling vertices with flag comp_n
        :param v0: Start vertex
        :param comp_n: Flag to label
        :return: None
        """
        q = [v0]
        d = {v0: 0}
        self.used[v0] = comp_n
        c_count = 1
        # c_depth = -1
        while len(q) != 0:
            v = q.pop(0)
            for ue in self.g[v]:
                u = ue[0]
                if self.used[u] == -1:
                    d[u] = d[v] + 1
                    c_count += 1
                    '''
                    # Depth-count condition
                    if c_count >= self.lim and c_depth == -1:
                        c_depth = d[u]
                    elif c_depth != -1 and c_depth < d[u]:
                        return
                    '''
                    # Count only condition
                    if c_count > self.lim:
                        return
                    q.append(u)
                    self.used[u] = comp_n

    def get_list(self) -> List[Dict[int, List[Tuple[int, int, Union[float, None]]]]]:
        """
        Returns the components adjacency lists with additional info
        :return: Adjacency lists
        """
        res = [{} for _ in range(self.sz)]

        for v, _ in enumerate(self.g):
            for ue in self.g[v]:
                u = ue[0]
                lv = self.used[v]
                lu = self.used[u]
                if lv == lu:
                    continue

                # res[lv].setdefault(lu, []).append((v, ) + ue)
                res[lv].setdefault(lu, []).extend(ue[1])

        return res

    def get_comp_to_vertex_mapping(self) -> Dict[int, List[int]]:
        """
        Returns new component -> vertices mapping
        :return: Dictionary with mapping
        """
        res = {}
        for v, comp in enumerate(self.used):
            res.setdefault(comp, []).append(v)
        return res


def simplify_graph(g: DWGraph) -> List[Dict[int, List[Tuple[int, int, Union[float, None]]]]]:
    """
    Returns graph in form of basic structures
    :param g: Graph
    :return: Adjacency lists with additional info
    """
    sg = [{} for _ in g.nodes]

    for v, u, w in g.edges.data('weight'):
        sg[v].setdefault(u, []).append((v, u, w))

    return sg


def simple_labels(g: DWGraph) -> Tuple[DWGraph, List[int], Dict[int, int]]:
    """
    Rename nodes of graph G with consecutive integers
    :param g: Graph G
    :return: Modified graph, mapping new -> old, mapping old -> new
    """
    nodes = list(g.nodes)
    rev_nodes_dict = {}
    for i, v in enumerate(nodes):
        rev_nodes_dict[v] = i

    g1 = nx.relabel_nodes(g, rev_nodes_dict)
    lab = g.get_labels()
    g1._labels = {i: lab[nodes[i]] for i in range(len(nodes))}

    return g1, nodes, rev_nodes_dict


def reverse_list(l: List[int]) -> Dict[int, int]:
    """
    Reverses the bijective mapping which is represented with list
    :param l: List to reverse
    :return: Mapping dictionary
    """
    d = {}
    for i, el in enumerate(l):
        d[el] = i
    return d


def is_tuple_ok(tu: Tuple[int], tv: Tuple[int], g: DWGraph,
                g_labels: Dict[int, int],
                edges_between: List[Tuple[int, int, Union[float, None]]],
                cu_reversed: Dict[int, int], cv_reversed: Dict[int, int],
                g0_labels: Dict[int, int],
                g0_nodes: List[int], edges_matching: DWGraph.EdgesMatchingType) -> bool:
    """
    May these tuples be concatenated?
    :param tu: Tuple for component U
    :param tv: Tuple for component V
    :param g: Background graph
    :param g_labels: Labels of G
    :param edges_between: Edges between U and V
    :param cu_reversed: Reversed vertices dictionary for component U
    :param cv_reversed: Reversed vertices dictionary for component U
    :param g0_labels: Labels of non-modified graph G0
    :param g0_nodes: List of matching new vertices to non-modified
    :param edges_matching: Edges matching function (may be zero)
    :return: True if these tuples may be concatenated, False otherwise
    """
    for (u, v, w) in edges_between:
        ui = cu_reversed[u]
        vi = cv_reversed[v]
        x = tu[ui]
        y = tv[vi]

        template_edge = (g0_labels[g0_nodes[u]], g0_labels[g0_nodes[v]], w)
        x_lab = g_labels[x]
        y_lab = g_labels[y]

        matched = False
        for edge_data in g[x][y].values():
            if edges_matching is None:
                if x_lab == template_edge[0] and y_lab == template_edge[1]:
                    matched = True
                    break
            else:
                g_w = edge_data['weight']
                if edges_matching(template_edge, (x_lab, y_lab, g_w)):
                    matched = True
                    break

        if not matched:
            return False

    return True


def recursive_tuples_step(sg0_list: List[Dict[int, List[Tuple[int, int, Union[float, None]]]]],
                          g0_tuples_list: List[List[Tuple[int]]],
                          g0_bfs_mapping: List[List[int]],
                          g: DWGraph,
                          g_labels: Dict[int, int],
                          g0_labels: Dict[int, int],
                          g0_nodes: List[int],
                          edges_matching: DWGraph.EdgesMatchingType,
                          comp_approx: int)\
        -> Tuple[List[Dict[int, List[Tuple[int, int, Union[float, None]]]]],
                 List[List[Tuple[int]]],
                 List[List[int]]]:
    """
    Makes one step of tuples matching

    :param sg0_list: Components graph with lists of G0 edges matched to each edge
    :param g0_tuples_list: List of matching tuples for each component
    :param g0_bfs_mapping: List of vertices of G0 matched to each component
    :param g: Background graph
    :param g_labels: Labels of graph G
    :param g0_labels: Labels of non-modified graph G0
    :param g0_nodes: List of matching new vertices to non-modified
    :param edges_matching: Edges matching function (may be zero)
    :param comp_approx: Number of vertices in each component
    :return: Three first structures modified
    """

    components_simplified = [list(zip(list(comp_el.keys()),
                                      [lst for to_comp, lst in comp_el.items()])) for comp_el in sg0_list]
    components_nx_graph = nx.DiGraph()
    for u, edges in enumerate(components_simplified):
        for v, w in edges:
            components_nx_graph.add_edge(u, v)
    components_bfs = BFS(components_simplified, comp_approx)

    components_bfs_mapping = components_bfs.get_comp_to_vertex_mapping()

    meta_tuples_list = [[] for _ in components_bfs_mapping]
    meta_bfs_mapping = [[] for _ in components_bfs_mapping]
    meta_components_list = components_bfs.get_list()

    for meta_component, meta_comp_vertices in components_bfs_mapping.items():
        meta_comp_subgraph = nx.subgraph(components_nx_graph, meta_comp_vertices)
        comp_simple_uf = {v: v for v in meta_comp_vertices}
        for u_comp, v_comp in meta_comp_subgraph.edges:
            gu = comp_simple_uf[u_comp]
            gv = comp_simple_uf[v_comp]

            if gu == gv:
                continue

            edges_between = sg0_list[gu][gv]
            rev_edges = [] if gu not in sg0_list[gv] else sg0_list[gv][gu]
            cu_tuples = g0_tuples_list[gu]
            cv_tuples = g0_tuples_list[gv]
            cu_vertices = g0_bfs_mapping[gu]
            cv_vertices = g0_bfs_mapping[gv]

            cu_reversed = reverse_list(cu_vertices)
            cv_reversed = reverse_list(cv_vertices)

            # Union the tuple lists
            new_tuples = []
            for tu in cu_tuples:
                for tv in cv_tuples:
                    if set(tu).intersection(set(tv)):
                        continue
                    t = tu + tv

                    tuple_ok = is_tuple_ok(tu, tv, g, g_labels, edges_between, cu_reversed, cv_reversed,
                                           g0_labels, g0_nodes, edges_matching)
                    if tuple_ok:
                        tuple_ok = is_tuple_ok(tv, tu, g, g_labels, rev_edges, cv_reversed, cu_reversed,
                                               g0_labels, g0_nodes, edges_matching)
                    if tuple_ok:
                        new_tuples.append(t)

            g0_tuples_list[gv] = new_tuples
            g0_bfs_mapping[gv] = cu_vertices + cv_vertices
            for c, c_list in sg0_list[gu].items():
                if c != gv:
                    sg0_list[gv].setdefault(c, []).extend(c_list)

            if gu in sg0_list[gv]:
                del sg0_list[gv][gu]

            g0_tuples_list[gu] = []
            g0_bfs_mapping[gu] = []
            sg0_list[gu] = {}

            for _k, _ in enumerate(sg0_list):
                for _uk in list(sg0_list[_k].keys()):
                    if _uk == gu:
                        sg0_list[_k].setdefault(gv, []).extend(sg0_list[_k][gu])
                        del sg0_list[_k][gu]

            for q, c in list(comp_simple_uf.items()):
                if c == gu:
                    comp_simple_uf[q] = gv

        new_meta_label = comp_simple_uf[meta_comp_vertices[0]]
        meta_tuples_list[meta_component] = g0_tuples_list[new_meta_label]
        meta_bfs_mapping[meta_component] = g0_bfs_mapping[new_meta_label]

    return meta_components_list, meta_tuples_list, meta_bfs_mapping


def recursive_unique_tuples(g: DWGraph, _g0: DWGraph,
                            fr: Dict[int, Set[int]],
                            g0_ordered_vertices: List[int],
                            edges_matching: DWGraph.EdgesMatchingType,
                            comp_approx: int=10) -> List[Tuple[int]]:
    """
    Matches the tuples of vertices of background graph to this template graph
    :param g: Background graph G
    :param _g0: Template graph G0
    :param fr: Multi-mapping fr: V0 -> V
    :param g0_ordered_vertices: List of vertices in G0 (in order to match them)
    :param edges_matching: Edges matching function (may be zero)
    :param comp_approx: Number of vertices in each component
    :return: List of matched tuples
    """
    g_labels = g.get_labels()
    g0_labels = _g0.get_labels()

    # First we want to relabel nodes with integers 0, 1, 2, ...
    g0, g0_nodes, g0_rev_nodes = simple_labels(_g0)

    sg0_list = simplify_graph(g0)
    g0_bfs_mapping = [[q] for q in range(len(g0_nodes))]
    g0_tuples_list = [[(v, ) for v in fr[g0_nodes[q]]] for q in range(len(g0_nodes))]

    prev_len = -1

    while True:
        if len(g0_bfs_mapping) <= 1 or len(g0_bfs_mapping) == prev_len:
            break
        prev_len = len(g0_bfs_mapping)
        print('Number of components: {}'.format(prev_len))
        sg0_list, g0_tuples_list, g0_bfs_mapping =\
            recursive_tuples_step(sg0_list, g0_tuples_list, g0_bfs_mapping,
                                  g, g_labels, g0_labels, g0_nodes, edges_matching, comp_approx)

    resulting_tuples = g0_tuples_list[0]
    for t_list in g0_tuples_list[1:]:
        new_r = []
        for et in resulting_tuples:
            for t in t_list:
                new_r.append(et + t)
        resulting_tuples = new_r
    resulting_vertices = []
    for l in g0_bfs_mapping:
        resulting_vertices.extend(l)
    resulting_vertices_g0 = [g0_nodes[i] for i in resulting_vertices]

    perm = [resulting_vertices_g0.index(i) for i in g0_ordered_vertices]

    res = []
    for t in resulting_tuples:
        nt = tuple([t[i] for i in perm])
        res.append(nt)

    return res
