from run import run, run_config as v_elimination_demo
from config import Config
from information_content import demo as info_ratio_demo
import config_helper

# Change index in brackets to choose another function
run_config = (v_elimination_demo, info_ratio_demo)[0]


# First example from the article
def test_example1():
    run("example1")


# Example from the article with another background graph
def test_example2():
    config = Config.get_config("example1")
    config.background_file_path = "../data/ex2_base.graph"
    config.background_labeling_file = "../data/ex2_base.labels"
    run_config(config)


def test_example2_2():
    config = Config.get_config("example1")
    config.background_file_path = "../data/art_ex2.graph"
    config.background_labeling_file = "../data/art_ex2.labels"
    config.template_file_path = "../data/art_template2.graph"
    config.template_labeling_file = "../data/art_template2.labels"
    run_config(config)


# Example from the first page of the article
def test_example3():
    config = Config.get_config("example1")
    config.background_file_path = "../data/art_ex3.graph"
    config.background_labeling_file = "../data/art_ex3.labels"
    config.template_file_path = "../data/art_template3.graph"
    config.template_labeling_file = "../data/art_template3.labels"
    run_config(config)


# Example from the first page of the article
def test_example5():
    config = Config.get_config("example1")
    config.background_file_path = "../data/art_ex5.graph"
    config.background_labeling_file = "../data/art_ex5.labels"
    config.template_file_path = "../data/art_template5.graph"
    config.template_labeling_file = "../data/art_template5.labels"
    run_config(config)


# Example with your files
def test_big_example():
    run("default")


# Example with your files with making graph undirected
def test_big_example_simple():
    config = Config.get_config("default")
    config.background_keep_directed = False
    config.template_keep_directed = False
    run_config(config)


# Example with your files with making graph undirected
def test_big_example_simple2():
    config = Config.get_config("default")
    config.background_file_path = "../data/art_ex4.graph"
    config.template_file_path = "../data/art_template4.graph"
    config.output_vertices_lists = "../data/vertices_subsets.txt"
    config.output_subgraphs = "../data/subgraph_"
    # config.background_labeling_function = None
    # config.template_labeling_function = None
    config.background_keep_directed = False
    config.template_keep_directed = False
    config.template_skip_header = False
    run_config(config)


# Example with very big data
def test_very_big_example():
    config = Config.get_config("default")
    config.background_file_path = "../data/50K_G.wgraph"
    config.template_file_path = "../data/50K_G0.wgraph"
    # config.background_labeling_function = None
    # config.template_labeling_function = None
    config.background_keep_directed = False
    config.template_keep_directed = False
    config.template_skip_header = False
    config.background_skip_header = False
    config.background_labeling_function = config_helper.labeling_function_50k
    config.template_labeling_function = config_helper.labeling_function_50k

    config.output_t = "../data/T_vb.txt"
    config.output_vertices_lists = "../data/vertices_subsets_vb.txt"
    config.output_subgraphs = "../data/vb_subgraph_"

    run_config(config)


# Example with very big data
def test_very_big_example_2():
    config = Config.get_config("default")
    config.background_file_path = "../data/50K_G_n.edges"
    config.template_file_path = "../data/50K_G0_n.wgraph"
    # config.background_labeling_function = None
    # config.template_labeling_function = None
    config.background_keep_directed = True
    config.template_keep_directed = True
    config.template_skip_header = False
    config.background_skip_header = False
    config.background_labeling_function = config_helper.labeling_function_50k
    config.template_labeling_function = config_helper.labeling_function_50k

    config.output_t = "../data/T_vb_2.txt"
    config.output_vertices_lists = "../data/vertices_subsets_vb_2.txt"
    config.output_subgraphs = "../data/vb_subgraph2_"

    run_config(config)

if __name__ == "__main__":
    # test_example1()
    # test_example2_2()
    # test_example2()
    # test_example3()
    # test_big_example_simple()
    # test_big_example()
    # test_big_example_simple2()
    # test_very_big_example()
    test_very_big_example_2()
