from config_helper import *


class Config:
    def __init__(self):
        # Properties for background graph G
        # Path to file with edges list
        self.background_file_path = None
        # If the file has the header line, set this to True
        self.background_skip_header = None
        # If the graph is weighted and the file has the weights column, set this to True
        self.background_weighted = None
        # If you want to keep direction of the edges, set this to True,
        # or set this to False to make graph undirected.
        self.background_keep_directed = None
        # You may specify the function l: V -> L. The default behavior is l(v) = v
        self.background_labeling_function = None
        # You may also specify the file to read labels from.
        # Make sure that self.background_labeling_function is None if you want to use this option
        self.background_labeling_file = None

        # The same properties for template graph G0
        self.template_file_path = None
        self.template_skip_header = None
        self.template_weighted = None
        self.template_keep_directed = None
        self.template_labeling_function = None
        self.template_labeling_file = None

        # Matches two edges: (u, v) in E and (u0, v0) in E0
        # Takes two tuples: (l(u), l(v), w(u, v)) and (l(u0), l(v0), w(u0, v0))
        # Returns True if edges match
        self.edges_matching_function = None

        self.edges_typing = None
        self.edges_epsilon_match = None

        # Number of iterations in LCC
        self.lcc_max_iterations = None

        # Output options
        # File path to print out T set or None if you don't want to print it.
        self.output_t = None
        # File path to print out vertices subsets for each template entrance
        # or None if you don't want to print them.
        self.output_vertices_lists = None
        # File path prefix to print out subgraphs of G for each template entrance.
        # Note that number of created files is equal to number of vertices subsets.
        # File names will have format <prefix><index>.graph
        # Initialize this with None to skip making subgraphs.
        self.output_subgraphs = None

        # File paths to print info ratios dictionary (before and after making undirected)
        self.output_info_ratios_before = None
        self.output_info_ratios_after = None

        # Parameter for graph matching
        self.max_component_elements = None

    @staticmethod
    def get_config(name) -> 'Config':
        config = Config()
        exec("config."+name+"()")
        return config

    # The default configuration
    # To add another configuration, just add one more method to this class
    # It should set all the properties that set to None in __init__.
    # You may also change this configuration.
    def default(self):
        self.background_file_path = "../data/big_background.wgraph"
        self.background_skip_header = False
        self.background_weighted = True
        self.background_keep_directed = False
        self.background_labeling_function = example_labeling_function

        self.template_file_path = "../data/big_template.wgraph"
        self.template_skip_header = True
        self.template_weighted = True
        self.template_keep_directed = False
        self.template_labeling_function = example_labeling_function

        # Define epsilon
        eps = 1e-10
        edges_matcher = EdgesMatcher(eps)
        self.edges_matching_function = edges_matcher.e_matching
        self.edges_typing = EdgesMatcher.edge_type
        self.edges_epsilon_match = eps

        self.lcc_max_iterations = 10
        self.max_component_elements = 10

        self.output_t = "../data/T.txt"
        self.output_vertices_lists = "../data/vertices_subsets_big.txt"
        self.output_subgraphs = "../data/subgraph_"

        self.output_info_ratios_before = "../data/ratios_before.txt"
        self.output_info_ratios_after = "../data/ratios_after.txt"

    # First example from article
    def example1(self):
        self.background_file_path = "../data/art_ex1.graph"
        self.background_skip_header = False
        self.background_weighted = False
        self.background_keep_directed = False
        self.background_labeling_file = "../data/art_ex1.labels"

        self.template_file_path = "../data/art_template1.graph"
        self.template_skip_header = False
        self.template_weighted = False
        self.template_keep_directed = False
        self.template_labeling_file = "../data/art_template1.labels"

        self.edges_matching_function = None

        self.lcc_max_iterations = 2
        self.max_component_elements = 2

        self.output_t = "../data/T.txt"
        self.output_vertices_lists = "../data/vertices_subsets.txt"
        self.output_subgraphs = "../data/subgraph_"

        self.output_info_ratios_before = "../data/ratios_before.txt"
        self.output_info_ratios_after = "../data/ratios_after.txt"
