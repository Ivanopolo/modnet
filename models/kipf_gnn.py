import time

import networkx as nx
import tensorflow as tf
from kipf import utils as kipf_utils, models as kipf_models

from models import utils
from models.base_model import BaseModel


class KipfModularityNet(BaseModel):
    def __init__(self,
                 associative_communities=True,
                 epochs=100,
                 dropout=0.0,
                 learning_rate_schedule=(0.1,),
                 hidden1=50,
                 bethe_hessian_init=False,
                 verbose=True):
        self.associative_communities = associative_communities
        self.epochs = epochs
        self.dropout = dropout
        self.learning_rate_schedule = learning_rate_schedule
        self.hidden1 = hidden1
        self.bethe_hessian_init = bethe_hessian_init
        self.verbose = verbose

    def fit_transform(self, graph):
        num_communities = len(graph.graph['partition'])
        adjacency_matrix = nx.adjacency_matrix(graph)
        support = [kipf_utils.preprocess_adj(adjacency_matrix)]
        num_supports = 1

        placeholders = {
            'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
            'dropout': tf.placeholder_with_default(0., shape=())
        }

        features = None
        if self.bethe_hessian_init:
            features = utils.bethe_hessian_node_features(
                adjacency_matrix,
                num_communities,
                self.associative_communities)

            placeholders['features'] = tf.placeholder(tf.float32)

        # Create model
        input_dim = adjacency_matrix.shape[0] if features is None else num_communities
        model = kipf_models.AdaptedGCN(placeholders,
                           input_dim=input_dim,
                           num_communities=num_communities,
                           hidden1=self.hidden1,
                           logging=True)

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

        feed_dict = {}
        feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
        feed_dict.update({placeholders['dropout']: self.dropout})
        if self.bethe_hessian_init:
            feed_dict.update({placeholders['features']: features})

        # Train model
        for lr in self.learning_rate_schedule:
            feed_dict.update({lr_ph: lr})
            for epoch in range(self.epochs):
                t = time.time()
                outs = sess.run([opt_op, loss], feed_dict)
                curr_loss = outs[1][0][0]

                if epoch > 10 and prev_loss - curr_loss <= 1e-5:
                    break

                if self.verbose:
                    print("Epoch:", '%04d' % (epoch + 1), "modularity_loss=", "{:.5f}".format(-curr_loss),
                          "time=", "{:.5f}".format(time.time() - t), "lr=", "{:.3f}".format(lr))

                prev_loss = curr_loss

        marginals = sess.run([model.predict()], feed_dict=feed_dict)[0]
        return marginals
