import numpy as np
from networkx.algorithms.community import greedy_modularity_communities

from models.base_model import BaseModel


class GreedyModularityModel(BaseModel):
    def fit_transform(self, graph):
        partition = list(greedy_modularity_communities(graph))
        num_communities_detected = len(partition)
        marginals = np.zeros([len(graph), num_communities_detected], dtype=np.float32)
        for i, nodes in enumerate(partition):
            marginals[list(nodes), i] = 1.0
        return marginals
