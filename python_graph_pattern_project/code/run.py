from config import Config
from DirectedWeightedGraph import DWGraph
import sys
from time import clock


def run_config(c: Config):
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

    # Perform elimination
    t = DWGraph.vertex_elimination_r(g, g0, c.lcc_max_iterations, c.edges_matching_function)
    if t is None:
        print("Template is not found")
        t = {}

    make_vertices = c.output_vertices_lists is not None or c.output_subgraphs is not None

    if c.output_t is not None:
        with open(c.output_t, 'w') as f:
            # Print the subset we've got
            f.write(str(t))
            # Print length of the subset
            # f.write("\n|T| = " + str(len(t)))

    vertices_list = None
    if make_vertices:
        # Get list of vertices

        start = clock()

        # vertices_list = DWGraph.get_vertices_list(g, g0, t, c.output_vertices_lists, c.edges_matching_function)

        vertices_list = DWGraph.get_vertices_list_recursive(g, g0, t,
                                                            c.output_vertices_lists,
                                                            c.edges_matching_function,
                                                            c.max_component_elements)

        end = clock()
        print("Time spent on making vertices list: {} s.".format(end - start))

    if c.output_vertices_lists is not None:
        pass
        # with open(c.output_vertices_lists, 'w') as f:
        # Print the list of vertices
        # f.write(str(vertices_list))
        # pass

    if c.output_subgraphs is not None:
        # Get list of subgraphs
        g_subgraphs = DWGraph.get_graphs_by_vertices_list(g0, vertices_list)
        # Save all the subgraphs
        for i, subgraph in enumerate(g_subgraphs):
            subgraph.write_to_file(c.output_subgraphs + str(i) + ".graph")


def run(config_name='default'):
    try:
        c = Config.get_config(config_name)
        run_config(c)

    except NameError:
        exit("There is no such configuration defined in config.py file.")


if __name__ == "__main__":
    n = len(sys.argv)
    c_name = None
    if n == 1:
        c_name = 'default'
    elif n == 2:
        c_name = sys.argv[1]
    else:
        exit("Specify only one argument or no arguments if you want to run default configuration.")

    run(c_name)
