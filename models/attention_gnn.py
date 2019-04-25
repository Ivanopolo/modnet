import graph_nets as gn
import networkx as nx
import numpy as np
import scipy
import tensorflow as tf
from scipy.sparse.linalg import eigsh

from models import utils
from models.attention_gnn_modules import AttentionGNN
from models.base_net_model import BaseNet


class AttentionModularityNet(BaseNet):
    def __init__(self,
                 associative=True,
                 epochs=150,
                 learning_rate=0.01,
                 num_heads=3,
                 num_layers=2,
                 hidden1=16*3,
                 emb_size=5,
                 lam=0.5,
                 bethe_hessian_init=False,
                 verbose=False):
        super(AttentionModularityNet, self).__init__(
            associative=associative, epochs=epochs, learning_rate=learning_rate, lam=lam, verbose=verbose)

        self.associative = associative
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden1 = hidden1
        self.emb_size = emb_size
        self.lam = lam
        self.bethe_hessian_init = bethe_hessian_init
        self.verbose = verbose

    @staticmethod
    def graph2graph_tuple(graph_nx, node_features):
        graph = graph_nx.to_directed()
        m = len(graph.edges)

        n = len(graph.nodes)
        senders, receivers = zip(*graph.edges)
        senders = tf.constant(np.array(senders, dtype=np.int32), tf.int32, name="senders")
        receivers = tf.constant(np.array(receivers, dtype=np.int32), tf.int32, name="receivers")

        dt = {
            "nodes": node_features if node_features is not None else [None for _ in range(n)],
            "edges": None,
            "receivers": receivers,
            "senders": senders,
            "globals": None,
            "n_node": n,
            "n_edge": m
        }

        return gn.graphs.GraphsTuple(**dt)

    def graph_batch2graph_tuple(self, adjs, graphs):
        graph_dicts = []
        for adj, graph in zip(adjs, graphs):
            bethe_hessian = utils.build_unweighted_bethe_hessian(adj, associative=self.associative)
            vals, node_vecs = scipy.sparse.linalg.eigsh(bethe_hessian, k=self.emb_size, which="SA")
            n = len(graph.nodes)
            senders, receivers = zip(*graph.edges)
            senders = np.array(senders, dtype=np.int32)
            receivers = np.array(receivers, dtype=np.int32)

            dt = {
                "nodes": node_vecs.astype(np.float32) if self.bethe_hessian_init else [None for _ in range(n)],
                "edges": None,
                "receivers": receivers,
                "senders": senders,
                "globals": None
            }
            graph_dicts.append(dt)

        return gn.utils_np.data_dicts_to_graphs_tuple(graph_dicts)

    def fit_transform(self, graph):
        num_communities = len(graph.graph['partition'])
        adjacency_matrix = nx.adjacency_matrix(graph)
        n = adjacency_matrix.shape[0]
        model = AttentionGNN(
            num_classes=num_communities,
            num_nodes=n,
            emb_size=self.emb_size,
            hidden=self.hidden1,
            num_heads=self.num_heads,
            num_iters=self.num_layers)

        features = None
        if self.bethe_hessian_init:
            features = utils.bethe_hessian_node_features(
                adjacency_matrix,
                self.emb_size,
                self.associative)
        graph_tuple = self.graph2graph_tuple(graph, features)
        model(graph_tuple)
        proba = model.predict()
        return self._fit_transform(graph, proba)
