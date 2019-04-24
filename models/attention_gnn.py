import time

import graph_nets as gn
import networkx as nx
import numpy as np
import scipy
import tensorflow as tf
from scipy.sparse.linalg import eigsh

from models import utils
from models.attention_gnn_modules import AttentionGNN
from models.base_model import BaseModel


class AttentionModularityNet(BaseModel):
    def __init__(self,
                 associative_communities=True,
                 epochs=150,
                 learning_rate=0.01,
                 num_heads=3,
                 num_layers=2,
                 hidden1=16*3,
                 emb_size=5,
                 lam=0.3,
                 bethe_hessian_init=False,
                 verbose=True):
        self.associative_communities = associative_communities
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
            "nodes": node_features,
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
            bethe_hessian = utils.build_unweighted_bethe_hessian(adj, associative=self.associative_communities)
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

    def fit_transform_batch(self, adjs, graphs):
        num_graphs = len(graphs)
        graph_size = adjs[0].shape[0]
        graph_tuple = self.graph_batch2graph_tuple(adjs, graphs)
        num_communities = len(graphs[0].graph['partition'])
        n = graph_size * num_graphs
        model = AttentionGNN(
            num_classes=num_communities,
            num_nodes=n,
            emb_size=self.emb_size,
            hidden=self.hidden1,
            num_heads=self.num_heads,
            num_iters=self.num_layers)

        model(graph_tuple)
        proba = model.predict()
        modularity = 0
        for i in range(num_graphs):
            modularity += utils.tf_modularity(
                adjs[i],
                tf.slice(proba, [i * graph_size, 0], [graph_size, num_communities]))
        modularity /= num_graphs

        equal_communities = 0
        community_ratio = 1.0 / num_communities
        for i in range(num_graphs):
            proba_slice = tf.slice(proba, [i * graph_size, 0], [graph_size, num_communities])
            deviation = tf.reduce_sum(
                (tf.reduce_sum(proba_slice, axis=0) / graph_size - community_ratio) ** 2
            )
            equal_communities += deviation
        equal_communities /= num_graphs

        mod_loss = -modularity if self.associative_communities else modularity
        loss = mod_loss + self.lam * equal_communities

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.learning_rate, global_step,
                                                   10, 0.96, staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        opt_op = optimizer.minimize(loss, global_step=global_step)

        # Initialize session
        sess = tf.Session()

        # Init variables
        sess.run(tf.global_variables_initializer())
        prev_loss = float('inf')

        # Train model
        for epoch in range(self.epochs):
            t = time.time()
            outs = sess.run([opt_op, mod_loss])
            curr_loss = outs[1][0][0]

            if self.verbose:
                print("Epoch:", '%04d' % (epoch + 1), "modularity_loss=", "{:.5f}".format(-curr_loss),
                      "time=", "{:.5f}".format(time.time() - t))

        marginals = sess.run([model.predict()])[0]
        print(marginals.sum(axis=0).astype(int))
        return marginals

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
                self.associative_communities)
        graph_tuple = self.graph2graph_tuple(graph, features)

        model(graph_tuple)
        proba = model.predict()
        modularity = utils.tf_modularity(adjacency_matrix, proba)
        loss = -modularity if self.associative_communities else modularity

        lr_ph = tf.placeholder(tf.float32)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_ph)
        opt_op = optimizer.minimize(loss)

        # Initialize session
        sess = tf.Session()

        # Init variables
        sess.run(tf.global_variables_initializer())
        prev_loss = float('inf')

        # Train model
        for lr in [self.learning_rate]:
            feed_dict = {lr_ph: lr}
            for epoch in range(self.epochs):
                t = time.time()
                outs = sess.run([opt_op, loss], feed_dict)
                curr_loss = outs[1][0][0]

                if self.verbose:
                    print("Epoch:", '%04d' % (epoch + 1), "modularity_loss=", "{:.5f}".format(-curr_loss),
                          "time=", "{:.5f}".format(time.time() - t), "lr=", "{:.3f}".format(lr))

        marginals = sess.run([model.predict()])[0]
        return marginals
