import os
import time
import glob

import numpy as np
import pandas as pd
from docopt import docopt
from sklearn.preprocessing import OneHotEncoder

import metrics
import models
from dataset import utils as dataset_utils


def main():
    args = docopt("""
    Usage:
        evaluate_models.py [options] <graphs_path>

    Options:
        --disassociative    Input graphs are disassociative
    """)

    graphs_path = args["<graphs_path>"]
    associative = not args["--disassociative"]

    metrics_order = {
        'Modularity': 0,
        'Soft Overlap': 1,
        'Hard Overlap': 2,
        'Mutual Information': 3
    }

    all_models = [
        models.TruePartitionModel(),
        models.BetheHessianModel(associative_communities=associative),
        models.KipfModularityNet(
            associative=associative,
            bethe_hessian_init=False,
            verbose=False),
        models.KipfModularityNet(
            associative=associative,
            bethe_hessian_init=True,
            verbose=False),
        models.AttentionModularityNet(
            associative=associative,
            bethe_hessian_init=False,
            verbose=False
        ),
        models.AttentionModularityNet(
            associative=associative,
            bethe_hessian_init=True,
            verbose=False
        )
    ]

    if associative:
        all_models.extend([
            models.GreedyModularityModel(),
            models.LouvainModularityModel(),
        ])

    num_test_graphs = len(glob.glob(os.path.join(graphs_path, "adj-*.npy")))
    print(f"Number of graphs found: {num_test_graphs}")

    output_path = os.path.join(graphs_path, "results")
    os.makedirs(output_path, exist_ok=True)

    results = np.zeros([num_test_graphs, len(all_models), len(metrics_order)])
    label_encoder = OneHotEncoder(categories='auto', sparse=False)
    print(time.time())

    for idx in range(num_test_graphs):
        if idx and idx % 10 == 0:
            print(idx, time.time())
            print_aggregated_results(results[:idx], metrics_order, all_models)
            np.save(os.path.join(output_path, f"{idx}.npy"), results[:idx])

        labels_path = os.path.join(graphs_path, f"labels-{idx}.npy")
        adj_path = os.path.join(graphs_path, f"adj-{idx}.npy")
        labels = np.load(labels_path, allow_pickle=True)
        true_labels = label_encoder.fit_transform(labels.reshape(-1, 1))
        all_models[0].true_labels = true_labels
        adjacency = np.load(adj_path, allow_pickle=True).tolist()
        graph_nx = dataset_utils.graph_from_adjacency(adjacency, labels)

        for j, m in enumerate(all_models):
            predictions = m.fit_transform(graph_nx)
            model_res = metrics.compute_all(graph_nx, true_labels, predictions)

            for metric, k in metrics_order.items():
                results[idx, j, k] = model_res[metric] if metric is not None else 0

    print_aggregated_results(results, metrics_order, all_models)
    np.save(os.path.join(output_path, f"all.npy"), results)


def print_aggregated_results(results, metrics_order, all_models):
    means = results.mean(axis=0)
    dt_means = {metric: means[:, order] for metric, order in metrics_order.items()}
    index = [type(m).__name__ for m in all_models]
    print(pd.DataFrame(dt_means, index=index).round(2))

if __name__ == '__main__':
    main()
