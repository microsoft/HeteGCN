import os
import numpy as np
import pandas as pd
from utils import normalize_sparse_graph, sparse_to_tuple
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize


class Dataset:
    """
    This class contains functions to prepare the data in the required format for consumption.
        1. Load Data
        2. Prepare Relevant Graphs
        3. Construct specifications for each HeteGCN layer
    """

    def __init__(self, args: dict):
        self.args = args.copy()
        self.dataset = self.args["dataset"]
        self.mount_dir = self.args.get("mount_dir", ".")
        self.path = self.args["path"]
        self.layers = self.path.split("-")

        self.data_dir = os.path.join(self.mount_dir, self.args["data_dir"])

        # If extended_data_dir is set, the training data is updated
        # This is used in low labelled settings.
        if self.args.get("extended_data_dir", None):
            self.extended_data_dir = os.path.join(
                self.mount_dir, self.args["extended_data_dir"]
            )
        else:
            self.extended_data_dir = None

        # Check if initial features or pretrained embeddings should be used
        self.use_feature_embeddings = self.args.get("use_feature_embeddings", 0) == 1

        self._setup_data()

    def _setup_data(self):
        """
        The orchestrating driver function.
        """
        self._load_data()

        self.NF = self.data["NF"]
        self.n_classes = self.data["train_labels"].shape[1]
        self.n_features = self.data["NF"].shape[1]
        self.n_nodes = self.data["NF"].shape[0]

        self.args["n_features"] = self.n_features
        self.args["n_classes"] = self.n_classes
        self.args["output_dims"] = self.n_classes

        if "hidden_dims" in self.args:
            self.hidden_dims = self.args["hidden_dims"]

        self._prepare_graphs()

        if self.use_feature_embeddings:
            self._prepare_feature_embeddings()

        self._prepare_layer_specs()

        self._prepare_labelled_data()

        self.args["layers"] = self.layers

    def _load_data(self):
        """
        - Load train/val/test data, relevant graphs.
        - If extended_data_dir is set, the train data is overriden. This is
          used when working with small labelled data.
        """
        self.data = pd.read_pickle(os.path.join(self.data_dir, self.dataset + ".pkl"))
        if self.extended_data_dir is not None:
            self.extended_data = pd.read_pickle(
                os.path.join(self.extended_data_dir, self.dataset + ".pkl")
            )
            self.data.update(self.extended_data)

    def _prepare_graphs(self):
        """
        - Load or construct relevant graphs.
        - Each graph can be row/symmetric normalized based
          on the hyperparameters.
        """
        if "NN" in self.layers:
            self._prepare_NN()

        if "FF" in self.layers:
            self._prepare_FF()

        if "NF" in self.layers:
            self._prepare_NF()

        if "FN" in self.layers:
            self._prepare_FN()

    def _prepare_NN(self, n_neighbors=25, metric="cosine"):
        # As there is no link graph available,
        # we construct a nearest neighbor graph from the BoW features
        self.NN = kneighbors_graph(
            self.NF, n_neighbors, metric=metric, include_self=True
        )

        if self.args["NN_norm"] == "row":
            self.NN = normalize_sparse_graph(self.NN, -1.0, 0.0)
        elif self.args["NN_norm"] == "sym":
            self.NN = normalize_sparse_graph(self.NN, -0.5, -0.5)

        self.args["NN"] = sparse_to_tuple(self.NN)

    def _prepare_FF(self):
        # PMI based FF matrix
        self.FF = self.data["FF"].copy()
        if self.args["FF_norm"] == "row":
            self.FF = normalize_sparse_graph(self.FF, -1.0, 0.0)
        elif self.args["FF_norm"] == "sym":
            self.FF = normalize_sparse_graph(self.FF, -0.5, -0.5)
        self.args["FF"] = sparse_to_tuple(self.FF)

    def _prepare_NF(self):
        # BoW feature vector
        if self.args["NF_norm"] == "row":
            self.NF = normalize_sparse_graph(self.NF, -1.0, 0.0)
        elif self.args["NF_norm"] == "sym":
            self.NF = normalize_sparse_graph(self.NF, -0.5, -0.5)
        elif self.args["NF_norm"] == "unit_length":
            self.NF = normalize(self.NF, norm="l2", axis=1)
        self.args["NF"] = sparse_to_tuple(self.NF)

    def _prepare_FN(self):
        self.FN = self.data["NF"].T
        if self.args["FN_norm"] == "row":
            self.FN = normalize_sparse_graph(self.FN, -1.0, 0.0)
        elif self.args["FN_norm"] == "sym":
            self.FN = normalize_sparse_graph(self.FN, -0.5, -0.5)
        self.args["FN"] = sparse_to_tuple(self.FN)

    def _prepare_labelled_data(self):
        self.train_nodes = self.data["train_nodes"].astype(int).reshape(-1, 1)
        self.train_labels = self.data["train_labels"].astype(int)
        self.train_data = np.concatenate(
            [self.train_nodes, self.train_labels], axis=1
        ).astype(int)

        self.val_nodes = self.data["val_nodes"].astype(int).reshape(-1, 1).astype(int)
        self.val_labels = self.data["val_labels"].astype(int)
        self.val_data = np.concatenate(
            [self.val_nodes, self.val_labels], axis=1
        ).astype(int)

        self.test_nodes = self.data["test_nodes"].astype(int).reshape(-1, 1).astype(int)
        self.test_labels = self.data["test_labels"].astype(int)
        self.test_data = np.concatenate(
            [self.test_nodes, self.test_labels], axis=1
        ).astype(int)

        self.args["train_nodes"] = self.train_nodes
        self.args["val_nodes"] = self.val_nodes
        self.args["test_nodes"] = self.test_nodes

        self.args["train_labels"] = self.train_labels
        self.args["val_labels"] = self.val_labels
        self.args["test_labels"] = self.test_labels

        self.args["train_data"] = self.train_data
        self.args["val_data"] = self.val_data
        self.args["test_data"] = self.test_data

    def _prepare_layer_specs(self):
        """
        The function is used to prepare auxiliary specifications for each layer
        containing details about:
            1. input dims
            2. output dims
        A and X are specified in the model.
        """
        if self.use_feature_embeddings:
            self.input_dims = self.feature_embeddings.shape[1]
        else:
            if self.layers[0].endswith("F"):
                self.input_dims = self.n_features
            if self.layers[0].endswith("N"):
                self.input_dims = self.n_nodes

        n_layers = len(self.layers)
        layer_specs = {}
        for i, layer in enumerate(self.layers):
            specs_dict = {
                "seed": self.args["seed"],
                "input_dims": self.input_dims if i == 0 else self.hidden_dims,
                "output_dims": self.hidden_dims
                if i < (n_layers - 1)
                else self.n_classes,
            }
            layer_specs[layer + "_%d" % i] = specs_dict

        self.args["layer_specs"] = layer_specs

    def _prepare_feature_embeddings(self):
        """
        - Prepare features or pretrained embeddings if applicable.
        - There is an option to premultiply AX of the first layer.
            - Speeds up training.
        """
        emb_dir = os.path.join(self.mount_dir, self.args["feature_embeddings"])
        self.feature_embeddings = pd.read_pickle(emb_dir + "/embs.pkl")
        if self.args["premultiply"]:
            init_graph = eval("self.%s" % self.layers[0])
            self.feature_embeddings = init_graph @ self.feature_embeddings
            self.layers[0] = "I"

        self.args["feature_embeddings"] = self.feature_embeddings
        self.args["input_dims"] = self.feature_embeddings.shape[1]

    def get_args(self):
        return self.args.copy()
