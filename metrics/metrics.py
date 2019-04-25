import numpy as np
from networkx.algorithms.community import modularity
from sklearn.metrics import normalized_mutual_info_score
from sympy.utilities.iterables import multiset_permutations


def community_soft_overlap(true_labels, marginals, normalize=True) -> float:
    assert true_labels.shape == marginals.shape
    num_vertices = true_labels.shape[0]
    num_communities = true_labels.shape[1]
    connect = ((marginals.sum(axis=0) / num_vertices) ** 2).sum()
    max_overlap = -float('inf')
    for p in multiset_permutations(np.arange(num_communities)):
        overlap = (true_labels[:, p] * marginals).sum() / num_vertices
        if normalize:
            overlap = (overlap - connect) / (1 - connect)
        if overlap > max_overlap:
            max_overlap = overlap

    return max_overlap


def community_hard_overlap(true_labels, marginals, normalize=True) -> float:
    assert true_labels.shape == marginals.shape
    num_vertices = true_labels.shape[0]
    num_communities = true_labels.shape[1]
    hard_labels = np.argmax(marginals, axis=1)
    predictions = np.zeros_like(true_labels)
    predictions[np.arange(len(predictions)), hard_labels] = 1.0
    norm = 1.0 / num_communities
    max_overlap = -float('inf')
    for p in multiset_permutations(np.arange(num_communities)):
        overlap = (true_labels[:, p] * predictions).sum() / num_vertices
        if normalize:
            overlap = (overlap - norm) / (1 - norm)
        if overlap > max_overlap:
            max_overlap = overlap

    return max_overlap


def compute_all(graph, true_labels, marginals):
    normalized_overlap = None
    normalized_hard_overlap = None

    if true_labels.shape[1] == marginals.shape[1]:
        normalized_overlap = community_soft_overlap(true_labels, marginals)
        normalized_hard_overlap = community_hard_overlap(true_labels, marginals)

    hard_labels = np.argmax(marginals, axis=1)
    partition = []
    for i in range(hard_labels.max() + 1):
        indices = np.argwhere(hard_labels == i).flatten()
        partition.append(set(indices))

    hard_modularity = modularity(graph, partition)
    true_labels_categorical = np.argmax(true_labels, axis=1)
    nmi = normalized_mutual_info_score(true_labels_categorical, hard_labels, average_method='arithmetic')
    return {
        "Soft Overlap": normalized_overlap,
        "Hard Overlap": normalized_hard_overlap,
        "Modularity": hard_modularity,
        "Mutual Information": nmi
    }
