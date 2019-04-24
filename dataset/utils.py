import os

import networkx as nx
import numpy as np


def generate_ssbm_graphs(
        num_nodes: int,
        num_communities: int,
        num_graphs: int,
        avg_degree_inside: float,
        avg_degree_between: float,
        seed: int = 0) -> list:

    """ Generates graphs using Symmetric Stochastic Block Model (SSBM) with specified parameters
    num_nodes - number of nodes
    num_communities - number of communities (should be k >= 5)
    num_graphs - number of graphs to generate
    avg_degree_inside - average node degree inside communities
    avg_degree_between - average node degree between communities
    seed - seed used to generate graphs
    """
    community_size = num_nodes // num_communities
    probability_inside_community = avg_degree_inside / num_nodes
    probability_between_communities = avg_degree_between / num_nodes
    graphs = []
    for i in range(num_graphs):
        graph = nx.random_partition_graph(
            [community_size for _ in range(num_communities)],
            probability_inside_community,
            probability_between_communities,
            seed=seed + i)

        graphs.append(graph)

    return graphs


def get_node_community_labels(graph: nx.Graph) -> list:
    return [node_attributes['block'] for _, node_attributes in graph.nodes(data=True)]


if __name__ == '__main__':
    n = 400
    k = 5
    N_train = 1
    N_test = 1
    rep = N_train + N_test
    regime = 'disassociative'

    a, b = 0, 18 # disassociative case
    # a, b = 21, 2 # associative case

    graphs = generate_ssbm_graphs(n, k, rep, a, b)

    # test dataset
    for i, g in enumerate(graphs[:N_test]):
        folder = f"graphs/{regime}_n{n}_k{k}/test/"
        os.makedirs(folder, exist_ok=True)
        A = nx.adjacency_matrix(g)
        np.save(f"{folder}matrix-{i}.npy", A)
        labels = get_node_community_labels(g)
        print(labels)
        np.save(f"{folder}labels-{i}.npy", labels)

    # train dataset
    for i, g in enumerate(graphs[N_test:]):
        folder = f"graphs/{regime}_n{n}_k{k}/train/"
        os.makedirs(folder, exist_ok=True)
        A = nx.adjacency_matrix(g)
        np.save(f"{folder}matrix-{i}.npy", A)
        labels = get_node_community_labels(g)
        np.save(f"{folder}labels-{i}.npy", labels)
