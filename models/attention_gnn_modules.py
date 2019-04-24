import graph_nets as gn
import sonnet as snt
import tensorflow as tf
import numpy as np


class Head(snt.AbstractModule):
    def __init__(self, num_heads, hidden, name="head"):
        super(Head, self).__init__(name=name)
        self.num_heads = num_heads
        self.hidden = hidden

    def _build(self, inputs):
        linear_val = snt.Linear(self.hidden, name="linear_val", use_bias=False)
        linear_key = snt.Linear(self.hidden, name="linear_key", use_bias=False)
        linear_query = snt.Linear(self.hidden, name="linear_query", use_bias=False)

        # inputs - [nodes, emb_dim]
        # tf.split(...) - [nodes, num_heads, emb_dim/num_heads]
        vals = tf.transpose(tf.split(linear_val(inputs), self.num_heads, axis=1), [1, 0, 2])
        keys = tf.transpose(tf.split(linear_key(inputs), self.num_heads, axis=1), [1, 0, 2])
        queries = tf.transpose(tf.split(linear_query(inputs), self.num_heads, axis=1), [1, 0, 2])
        queries /= np.sqrt(self.hidden / self.num_heads)
        return vals, keys, queries


class GraphAttention(snt.AbstractModule):
    def __init__(self, hidden, num_heads, name="graph_attention"):
        super(GraphAttention, self).__init__(name=name)
        self.hidden = hidden
        self.num_heads = num_heads

    def _build(self, inputs, input_graph):
        head = Head(self.num_heads, self.hidden)
        attention = gn.modules.SelfAttention()
        output_proj = snt.Linear(self.hidden, use_bias=False)
        mlp = snt.nets.MLP([self.hidden, self.hidden])
        norm1 = snt.BatchNorm()
        norm2 = snt.BatchNorm()

        vals, keys, queries = head(inputs)
        attended_nodes = attention(vals, keys, queries, input_graph).nodes
        output_projected = output_proj(tf.reduce_sum(attended_nodes, axis=1))

        ### This skip-connection is non-orthodox
        output_projected = norm1(output_projected +
                                 tf.reshape(vals, [-1, self.hidden]),
                                 is_training=True)

        normalized = norm2(mlp(output_projected) + output_projected, is_training=True)
        return normalized


class AttentionGNN(snt.AbstractModule):
    def __init__(self, num_classes, num_nodes, emb_size, hidden, num_heads, num_iters, name="attention_gnn"):
        super(AttentionGNN, self).__init__(name=name)
        self.num_classes = num_classes
        self.num_nodes = num_nodes
        self.emb_size = emb_size
        self.hidden = hidden
        self.num_heads = num_heads
        self.num_iters = num_iters
        self.ids = tf.constant(list(range(num_nodes)), dtype=tf.int32)

    def _build(self, input_graph):
        embed = snt.Embed(self.num_nodes, self.emb_size)
        linear = snt.Linear(self.hidden)
        norm = snt.BatchNorm()
        mlp = snt.nets.MLP([self.hidden, self.hidden], activate_final=False)
        gnn_attentions = []
        for _ in range(self.num_iters):
            gnn_attention = GraphAttention(self.hidden, self.num_heads)
            gnn_attentions.append(gnn_attention)
        final = snt.Linear(self.num_classes)

        node_features = input_graph.nodes
        if node_features[0] is not None:
            embs = tf.constant(node_features, dtype=tf.float32)
            embs = tf.nn.relu(linear(embs))
            embs = norm(mlp(embs) + embs, is_training=True)
        else:
            embs = embed(self.ids)

        for i in range(self.num_iters):
            embs = gnn_attentions[i](embs, input_graph)

        logits = final(embs)
        self.outputs = logits

    def predict(self):
        return tf.nn.softmax(self.outputs)
