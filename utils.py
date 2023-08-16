import os
import random
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from sklearn.metrics import accuracy_score, f1_score


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    # Borrowed From https://github.com/tkipf/gcn
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def set_seed(seed):
    """Sets seed for reproducability."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def normalize_sparse_graph(graph, gamma, beta):
    """
    Utility function for normalizing sparse graphs.
    return Dr^gamma x graph x Dc^beta
    """
    b_graph = graph.tocsr().copy()
    r_graph = b_graph.copy()
    c_graph = b_graph.copy()
    row_sums = []
    for i in range(graph.shape[0]):
        row_sum = r_graph.data[r_graph.indptr[i] : r_graph.indptr[i + 1]].sum()
        if row_sum == 0:
            row_sums.append(0.0)
        else:
            row_sums.append(row_sum**gamma)

    c_graph = c_graph.tocsc()
    col_sums = []
    for i in range(graph.shape[1]):
        col_sum = c_graph.data[c_graph.indptr[i] : c_graph.indptr[i + 1]].sum()

        if col_sum == 0:
            col_sums.append(0.0)
        else:
            col_sums.append(col_sum**beta)

    for i in range(graph.shape[0]):
        if row_sums[i] != 0:
            b_graph.data[r_graph.indptr[i] : r_graph.indptr[i + 1]] *= row_sums[i]

    b_graph = b_graph.tocsc()
    for i in range(graph.shape[1]):
        if col_sums[i] != 0:
            b_graph.data[c_graph.indptr[i] : c_graph.indptr[i + 1]] *= col_sums[i]
    return b_graph


def get_metrics(preds, labels, ID=""):
    """Utility function to compute Accuracy, MicroF1 and Macro F1"""
    accuracy = accuracy_score(preds, labels)
    micro = f1_score(preds, labels, average="micro")
    macro = f1_score(preds, labels, average="macro")
    best_metrics = {
        "%sAccuracy" % ID: accuracy,
        "%sMicro" % ID: micro,
        "%sMacro" % ID: macro,
    }
    return best_metrics
