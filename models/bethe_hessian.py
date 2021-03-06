import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

from models import utils
from models.base_model import BaseModel


class BetheHessianModel(BaseModel):
    def __init__(self, associative_communities=True, tol=0):
        self.associative_communities = associative_communities
        self.tol = tol

    def fit_transform(self, graph):
        num_communities = len(graph.graph['partition'])
        adj = nx.adjacency_matrix(graph)
        bethe_hessian = utils.build_unweighted_bethe_hessian(adj, self.associative_communities)
        vals, vecs = eigsh(bethe_hessian, num_communities, which='SA', tol=self.tol)
        kmeans = KMeans(n_clusters=num_communities)
        kmeans.fit(vecs)

        hard_labels = kmeans.labels_
        predictions = np.zeros([adj.shape[0], num_communities])
        predictions[np.arange(len(predictions)), hard_labels] = 1.0
        return predictions
