import networkx as nx
import numpy as np


def generate_ssbm_graphs(
        num_nodes: int,
        num_communities: int,
        num_graphs: int,
        avg_degree_inside: float,
        avg_degree_between: float,
        seed: int = 0):

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
    for i in range(num_graphs):
        graph = nx.random_partition_graph(
            [community_size for _ in range(num_communities)],
            probability_inside_community,
            probability_between_communities,
            seed=seed + i)

        yield graph


def get_node_community_labels(graph: nx.Graph) -> list:
    return [node_attributes['block'] for _, node_attributes in graph.nodes(data=True)]


def graph_from_adjacency(adj, labels):
    graph_nx = nx.convert_matrix.from_scipy_sparse_matrix(adj)

    partition = []
    for i in range(labels.max() + 1):
        partition.append(set(np.argwhere(labels == i).flatten()))
    graph_nx.graph['partition'] = partition
    return graph_nx
