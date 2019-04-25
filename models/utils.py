import numpy as np
from sklearn.preprocessing import OneHotEncoder
import networkx as nx
import tensorflow as tf
import scipy.sparse


def graph2labels(graph):
    partition = graph.graph['partition']
    labels = np.zeros([len(graph), len(partition)])
    for i, p in enumerate(partition):
        labels[list(p), i] = 1

    return labels


def softmax_predictions2labels(preds: np.array):
    pred_labels = np.argmax(preds, axis=1)
    enc = OneHotEncoder(preds.shape[1])
    return np.asarray(enc.fit_transform(pred_labels.reshape(-1, 1)).todense())


def sparse_matrix2sparse_tensor(sparse):
    coo = sparse.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data.astype(np.float32), coo.shape)


def modularity_loss(proba, adj, associative):
    '''
    Modulariry = trace(U B U^T)/ ||W|| where B=W-mm^T/ ||W|| and ||W|| = sum of it's elements
    '''
    W = sparse_matrix2sparse_tensor(adj)
    w = adj.sum()
    m = tf.constant(np.asarray(adj.sum(axis=0)), name="m", dtype=tf.float32)
    modularity = tf.reduce_sum(tf.multiply(proba, tf.sparse.matmul(W, proba)))
    proba_m = tf.matmul(m, proba)
    modularity -= tf.matmul(proba_m, proba_m, adjoint_b=True) / w
    modularity /= w
    return -modularity if associative else modularity


def community_size_regularization(proba, num_nodes, num_communities):
    community_ratio = 1.0 / num_communities
    regularization_loss = tf.reduce_sum(
        (tf.reduce_sum(proba, axis=0) / num_nodes - community_ratio) ** 2
    )
    return regularization_loss


def build_unweighted_bethe_hessian(adjacency_matrix, associative=True):
    degrees = np.asarray(adjacency_matrix.sum(axis=1), dtype=np.float64).flatten()
    r = np.sqrt((degrees ** 2).mean() / degrees.mean() - 1)
    r = r if associative else -r
    n = adjacency_matrix.shape[0]
    eye = scipy.sparse.eye(n, dtype=np.float64)
    D = scipy.sparse.spdiags(degrees, [0], n, n, format='csr')
    bethe_hessian = (r**2-1)*eye-r*adjacency_matrix+D
    return bethe_hessian


def bethe_hessian_node_features(adjacency_matrix, num_communities, associative=True):
    bethe_hessian = build_unweighted_bethe_hessian(adjacency_matrix, associative)
    _, node_vecs = scipy.sparse.linalg.eigsh(bethe_hessian, k=num_communities, which="SA")
    return node_vecs