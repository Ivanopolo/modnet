import os

import networkx as nx
import numpy as np
from docopt import docopt

from dataset import utils as dataset_utils


def main():
    args = docopt("""
    Usage:
        generate_datasets.py [options]

    Options:
        --num_nodes NUM                 Number of nodes in generated graphs [default: 400]
        --num_communities NUM           Number of planted communities [default: 5]
        --num_graphs NUM                Number of graphs to generate [default: 1000]
        --test_ratio NUM                Context distribution smoothing [default: 0.2]
    """)

    num_nodes = int(args["--num_nodes"])
    num_communities = int(args["--num_communities"])
    num_graphs = int(args["--num_graphs"])
    test_ratio = float(args["--test_ratio"])
    train_size = int(num_graphs * (1-test_ratio))

    regimes = [
        ("associative", 21, 2),
        ("disassociative", 0, 18)
    ]

    for regime_name, avg_degree_inside, avg_degree_between in regimes:
        graphs_generator = dataset_utils.generate_ssbm_graphs(
            num_nodes, num_communities, num_graphs, avg_degree_inside, avg_degree_between)

        output_train_folder = f"graphs/{regime_name}_n={num_nodes}_k={num_communities}/train/"
        output_test_folder = f"graphs/{regime_name}_n={num_nodes}_k={num_communities}/test/"
        os.makedirs(output_train_folder, exist_ok=True)
        os.makedirs(output_test_folder, exist_ok=True)

        for i, graph in enumerate(graphs_generator):
            output_folder = output_train_folder if i < train_size else output_test_folder
            graph_index = i if i < train_size else i - train_size
            adjacency = nx.adjacency_matrix(graph)
            np.save(output_folder + f"adj-{graph_index}", adjacency)
            labels = dataset_utils.get_node_community_labels(graph)
            np.save(output_folder + f"labels-{graph_index}", labels)


if __name__ == '__main__':
    main()
