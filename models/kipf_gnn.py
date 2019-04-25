import networkx as nx
import numpy as np
import tensorflow as tf

from kipf import utils as kipf_utils, models as kipf_models
from models import utils
from models.base_net_model import BaseNet


class KipfModularityNet(BaseNet):
    def __init__(self,
                 associative=True,
                 epochs=150,
                 dropout=0.0,
                 learning_rate=0.1,
                 hidden1=50,
                 bethe_hessian_init=False,
                 lam=0.5,
                 verbose=False):
        super(KipfModularityNet, self).__init__(
            associative=associative, epochs=epochs, learning_rate=learning_rate, lam=lam, verbose=verbose)

        self.associative = associative
        self.epochs = epochs
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.hidden1 = hidden1
        self.bethe_hessian_init = bethe_hessian_init
        self.verbose = verbose

    def fit_transform(self, graph):
        num_communities = len(graph.graph['partition'])
        adjacency_matrix = nx.adjacency_matrix(graph)
        indices, values, shape = kipf_utils.preprocess_adj(adjacency_matrix)
        support = [tf.SparseTensor(indices, values.astype(np.float32), shape)]
        features = None
        if self.bethe_hessian_init:
            features = utils.bethe_hessian_node_features(
                adjacency_matrix,
                num_communities,
                self.associative).astype(np.float32)

        # Create model
        input_dim = adjacency_matrix.shape[0] if features is None else num_communities
        model = kipf_models.AdaptedGCN(input_dim=input_dim,
                           features=features,
                           num_communities=num_communities,
                           hidden1=self.hidden1,
                           support=support,
                           logging=True)

        proba = model.predict()
        return self._fit_transform(graph, proba)
