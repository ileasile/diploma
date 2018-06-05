from config import Config
from DirectedWeightedGraph import DWGraph
import sys


def get_graph(file_path, is_weighted, make_undirected, skip_first_line, labeling_function, labeling_file):
    g = DWGraph.from_file_edge_list(file_path,
                                    is_weighted=is_weighted,
                                    make_undirected=make_undirected,
                                    skip_first_line=skip_first_line)
    # Add labels to graph
    if labeling_function is not None:
        g.add_labels_by_identifiers(labeling_function)
    elif labeling_file is not None:
        g.add_labels_from_file(labeling_file)
    else:
        g.add_default_labels()

    return g


def print_info(info, file_path):
    with open(file_path, 'w') as f:
        for (u, v), data in info.items():
            if isinstance(data, float):
                f.write("{}\t{}\t{}\n".format(u, v, data))
            elif isinstance(data, dict):
                for w, r in data.items():
                    f.write("{}\t{}\t{}\t{}\n".format(u, v, w, r))
            else:
                exit("Bad info format.")


def demo(c: Config):
    # Before making undirected
    print("=== Before making undirected ===")
    print("Reading and labeling graphs...")
    g = get_graph(c.background_file_path, c.background_weighted, False, c.background_skip_header,
                  c.background_labeling_function, c.background_labeling_file)
    g0 = get_graph(c.template_file_path, c.template_weighted, False, c.template_skip_header,
                   c.template_labeling_function, c.template_labeling_file)

    info = DWGraph.get_ratios(g, g0, c.edges_typing, True, c.edges_typing, True, c.edges_epsilon_match)
    print_info(info, c.output_info_ratios_before)

    del g, g0, info

    # After making undirected
    print("=== After making undirected ===")
    print("Reading and labeling graphs...")
    g = get_graph(c.background_file_path, c.background_weighted, not c.background_keep_directed,
                  c.background_skip_header, c.background_labeling_function, c.background_labeling_file)
    g0 = get_graph(c.template_file_path, c.template_weighted, not c.background_keep_directed,
                   c.template_skip_header, c.template_labeling_function, c.template_labeling_file)

    info = DWGraph.get_ratios(g, g0, c.edges_typing, True, c.edges_typing, True, c.edges_epsilon_match)
    print_info(info, c.output_info_ratios_after)


def main_demo():
    n = len(sys.argv)
    c_name = None
    if n == 1:
        c_name = 'default'
    elif n == 2:
        c_name = sys.argv[1]
    else:
        exit("Specify only one argument or no arguments if you want to run default configuration.")

    try:
        c = Config.get_config(c_name)
        demo(c)

    except NameError:
        exit("There is no such configuration defined in config.py file.")


if __name__ == "__main__":
    main_demo()
