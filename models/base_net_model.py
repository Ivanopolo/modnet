import time

import networkx as nx
import tensorflow as tf

from models import utils
from models.base_model import BaseModel


class BaseNet(BaseModel):
    def __init__(self, associative, epochs, learning_rate, lam, verbose):
        self.associative = associative
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lam = lam
        self.verbose = verbose

    def _fit_transform(self, graph, predicted_probabilities):
        num_communities = len(graph.graph['partition'])
        adjacency_matrix = nx.adjacency_matrix(graph)
        mod_loss = utils.modularity_loss(predicted_probabilities, adjacency_matrix, self.associative)
        reg_loss = utils.community_size_regularization(predicted_probabilities, len(graph.nodes), num_communities)
        loss = mod_loss + self.lam * reg_loss

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(self.learning_rate, global_step,
                                                   10, 0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        opt_op = optimizer.minimize(loss)

        # Initialize session
        sess = tf.Session()

        # Init variables
        sess.run(tf.global_variables_initializer())

        # Train model
        for lr in [self.learning_rate]:
            for epoch in range(self.epochs):
                t = time.time()
                outs = sess.run([opt_op, mod_loss])
                curr_loss = outs[1][0][0]

                if self.verbose:
                    print("Epoch:", '%04d' % (epoch + 1), "modularity_loss=", "{:.5f}".format(-curr_loss),
                          "time=", "{:.5f}".format(time.time() - t), "lr=", "{:.3f}".format(lr))

        marginals = sess.run([predicted_probabilities])[0]
        return marginals

    def fit_transform(self, graph):
        raise NotImplementedError
