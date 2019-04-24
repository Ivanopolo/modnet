import community as louvain
import numpy as np

from models.base_model import BaseModel


class LouvainModularityModel(BaseModel):
    def __init__(self, resolution=1.0):
        self.resolution = resolution

    def fit_transform(self, graph):
        partition = louvain.best_partition(graph, resolution=self.resolution)
        num_communities_detected = len(set(partition.values()))
        marginals = np.zeros([len(graph), num_communities_detected], dtype=np.float32)
        for node, community in partition.items():
            marginals[node, community] = 1.0

        return marginals
