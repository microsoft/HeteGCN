import time
import numpy as np
import tensorflow as tf

from utils import get_metrics
from layers import GCN_Layer


class Model:
    """
    This class builds the HeteGCN model containing functions for the following:
        1. Define the sequence of propagation for the path.
        2. Build the tensorflow graph.
        3. Define the classification and the auxiliary losses.
        4. Define the placeholders and a mechanism to feed them.
        5. Training loop.
        6. Few other helper functions.
    """

    def __init__(self, args):
        self.args = args.copy()
        self.setup_placeholders()
        self.learning_rate = args["learning_rate"]

        self.nodes = self.args["nodes"]
        self.labels = self.args["labels"]

        self.layers = self.args["layers"]
        self.n_layers = len(self.layers)
        self.layer_specs = self.args["layer_specs"]

        self._build_graph()
        self._loss()

    def _build_graph(self):
        ####################################################
        # Used to extract intermediate outputs for analysis
        ####################################################
        self.model_weights = []
        self.layer_outputs = []
        self.aux_embeddings = []
        ####################################################

        # features set to None if no features are used in
        # in the first layer
        features = self.args.get("feature_embeddings", None)

        for i, layer in enumerate(self.layers):
            with tf.variable_scope(layer + "_%d" % i):
                # prepares the layer specification
                layer_specs_dict = self._get_layer_specs(layer + "_%d" % i, features)

                # HeteGCN Layer
                model = GCN_Layer(layer_specs_dict)

                # Query Layer Outputs
                features = model()

                # Apply the activation function for all layers except the final layers.
                if i < (self.n_layers - 1):
                    features = tf.nn.relu(features)

                ####################################################
                # Extracting the intermediate outputs
                ####################################################
                self.layer_outputs.append(features)
                self.model_weights.append(model.W)
                self.aux_embeddings.append(model.aux_embeddings)
                ####################################################

        self.all_predictions = features
        self.predictions = tf.gather(self.all_predictions, self.nodes)

    def _loss(self):
        """
        Total loss = Classification Loss + Weight Regularization + Embedding Regularization
            Classification Loss -> Cross Entropy Loss
        """
        self.emb_reg = self.args["emb_reg"] * self._emb_reg()
        self.wt_reg = self.args["wt_reg"] * self._wt_reg()
        self.pred_error = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.predictions, labels=self.labels
            )
        )
        self.loss = self.pred_error + self.wt_reg + self.emb_reg
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            self.loss
        )

    def _emb_reg(self):
        """
        If features or pretrained embeddings are used,
        embedding regularization is set to zero.
        """
        if "feature_embeddings" in self.placeholders:
            return tf.constant(0.0)
        return tf.reduce_mean(tf.square(tf.stack(tf.get_collection("embeddings"))))

    def _wt_reg(self):
        """
        Normalized Weight Regularization
        """
        l2_sum = tf.reduce_sum(
            [tf.reduce_sum(tf.square(w)) for w in tf.get_collection("weights")]
        )
        no_of_wts = tf.reduce_sum(
            tf.cast(tf.get_collection("#_of_weights"), tf.float32)
        )
        wt_reg = l2_sum / no_of_wts
        wt_reg = tf.where(tf.is_nan(wt_reg), tf.zeros_like(wt_reg), wt_reg)
        return wt_reg

    def setup_placeholders(self):
        """
        Define the required placeholders based on the path.
        """
        self.placeholders = {
            "nodes": tf.placeholder(tf.int32, shape=(None, 1), name="nodes"),
            "labels": tf.placeholder(
                tf.float32, shape=(None, self.args["output_dims"]), name="labels"
            ),
            "dropout": tf.placeholder_with_default(0.0, shape=(), name="dropout"),
            "learning_rate": tf.placeholder_with_default(
                0.0, shape=(), name="learning_rate"
            ),
        }

        if "NF" in self.args:
            self.placeholders.update(
                {"NF": tf.sparse_placeholder(tf.float32, name="NF")}
            )

        if "FN" in self.args:
            self.placeholders.update(
                {"FN": tf.sparse_placeholder(tf.float32, name="FN")}
            )

        if "NN" in self.args:
            self.placeholders.update(
                {"NN": tf.sparse_placeholder(tf.float32, name="NN")}
            )

        if "FF" in self.args:
            self.placeholders.update(
                {"FF": tf.sparse_placeholder(tf.float32, name="FF")}
            )

        if (
            "feature_embeddings" in self.args
            and not self.args["feature_embeddings"] is None
        ):
            self.placeholders.update(
                {
                    "feature_embeddings": tf.placeholder(
                        tf.float32,
                        shape=(self.args["n_features"], self.args["input_dims"]),
                        name="support_embeddings",
                    )
                }
            )

        self.args.update(self.placeholders)

    def construct_feed_dict(self, args):
        """
        Helper function to construct the feed dict.
        """
        feed_dict = {}
        for key, placeholder in self.placeholders.items():
            feed_dict.update({placeholder: args[key]})
        return feed_dict

    def _get_layer_specs(self, layer_ID, features=None):
        """
        Constructs the specification for each layer with the following details.
            1. A
            2. X
            3. Dropout
        """
        layer_specs_dict = self.layer_specs[layer_ID]
        layer_specs_dict["dropout"] = self.args["dropout"]
        layer_specs_dict["A"] = self.args.get(layer_ID.split("_")[0], None)
        if features is not None:
            layer_specs_dict["X"] = features
        return layer_specs_dict

    def fit(self, sess, args):
        """
        The training loop is defined in this function.
        """

        ####################################################
        # Setting up the data and auxiliary variables
        ####################################################
        learning_rate = args["learning_rate"]
        early_stopping = args["early_stopping"]
        decay_rate = args["decay_rate"]
        decay_freq = args["decay_freq"]
        epochs = args["epochs"]

        # Deep Copy
        train_data = args["train_data"].copy()
        val_data = args["val_data"].copy()
        test_data = args["test_data"].copy()

        # For Early Stopping
        patience = 0

        metrics = {}
        best_metrics = {}
        ####################################################

        # Training Loop
        for epoch in range(1, epochs + 1):
            ####################################################
            # Train
            ####################################################
            start_time = time.time()

            # Learning Rate Decay
            if epoch % decay_freq == 0:
                learning_rate = learning_rate * decay_rate

            # Shuffling the training Data
            np.random.shuffle(train_data)

            nodes = train_data[:, 0].reshape(-1, 1)
            labels = train_data[:, 1:]

            n_nodes = train_data.shape[0]
            batch_start = 0
            loss = 0
            pred_error = 0
            emb_reg = 0
            wt_reg = 0

            while batch_start < n_nodes:
                nodes_batch = nodes[
                    batch_start : min(n_nodes, batch_start + args["batch_size_train"])
                ]
                labels_batch = labels[
                    batch_start : min(n_nodes, batch_start + args["batch_size_train"])
                ]
                batch_start = batch_start + args["batch_size_train"]

                args["nodes"] = nodes_batch
                args["labels"] = labels_batch
                args["learning_rate"] = learning_rate

                feed_dict = self.construct_feed_dict(args)

                outs = sess.run(
                    [self.opt, self.loss, self.pred_error, self.emb_reg, self.wt_reg],
                    feed_dict=feed_dict,
                )
                loss += outs[1] * labels_batch.shape[0]
                pred_error += outs[2] * labels_batch.shape[0]
                emb_reg += outs[3] * labels_batch.shape[0]
                wt_reg += outs[4] * labels_batch.shape[0]

            loss = loss / n_nodes
            pred_error = pred_error / n_nodes
            emb_reg = emb_reg / n_nodes
            wt_reg = wt_reg / n_nodes
            ####################################################

            ####################################################
            # Evaluation
            ####################################################
            epoch_time = time.time() - start_time

            # Get Predictions for all the nodes
            all_predictions = self.get_predictions(sess, args)

            # Get Predicted Class Labels
            train_preds = all_predictions[train_data[:, 0]].argmax(axis=1)
            val_preds = all_predictions[val_data[:, 0]].argmax(axis=1)
            test_preds = all_predictions[test_data[:, 0]].argmax(axis=1)

            # Get True Class Labels
            y_train = train_data[:, 1:].argmax(axis=1)
            y_val = val_data[:, 1:].argmax(axis=1)
            y_test = test_data[:, 1:].argmax(axis=1)

            # Get metrics like accuracy, micro and macro scores.
            train_metrics = get_metrics(train_preds, y_train, "Train")
            val_metrics = get_metrics(val_preds, y_val, "Val")
            test_metrics = get_metrics(test_preds, y_test, "Test")

            metrics.update(train_metrics)
            metrics.update(val_metrics)
            metrics.update(test_metrics)
            ####################################################

            if args["verbose"]:
                print_str = "\nEpoch %03d - Loss: %0.06f  PredError: %0.06f  EmbReg: %0.06f  WtReg: %0.06f (%0.03fs)"
                print(
                    print_str % (epoch, loss, pred_error, emb_reg, wt_reg, epoch_time)
                )
                print("Train Acc - %0.03f" % metrics["TrainAccuracy"])
                print("Val Acc - %0.03f" % metrics["ValAccuracy"])
                print("Test Acc - %0.03f" % metrics["TestAccuracy"])

            if metrics["ValAccuracy"] > best_metrics.get("ValAccuracy", 0.0):
                best_metrics.update(metrics)
                dump = self.get_model_params(sess, args)
                patience = 0
            else:
                patience = patience + 1

            if patience == early_stopping:
                # Early Stopping
                break

        return best_metrics, dump

    def get_predictions(self, sess, args):
        """
        Gets predictions for all the nodes.
        """
        args["dropout"] = 0.0
        feed_dict = self.construct_feed_dict(args)
        all_predictions = sess.run(self.all_predictions, feed_dict=feed_dict)
        return all_predictions

    def get_model_params(self, sess, args):
        """
        Extracts various intermediate outputs from the model for analysis.
        """
        args["dropout"] = 0.0
        feed_dict = self.construct_feed_dict(args)
        model_weights = sess.run(self.model_weights, feed_dict=feed_dict)
        layer_outputs = sess.run(self.layer_outputs, feed_dict=feed_dict)
        aux_embeddings = sess.run(self.aux_embeddings, feed_dict=feed_dict)
        model_params = {
            "model_weights": model_weights,
            "layer_outputs": layer_outputs,
            "aux_embeddings": aux_embeddings,
        }
        return model_params
