# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Creates a Neural Decision tree classifier.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
import copy
from sklearn.cluster import AgglomerativeClustering

from art.utils import check_and_transform_label_format
from art.estimators.classification.classifier import Classifier

from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class NeuralDecisionTree(Classifier):
    """
    Implementation of Neural Decision Tree architecture used as part of an adapative robustness framework. A Neural Decision Tree 
    is composed of multiple classifiers in a tree structure. Each node in the tree classifies an input into a sub-group until  
    there is only one label in the sub-group, at which point the model's output is returned.
    | Paper link: https://arxiv.org/abs/2012.07887
    """

    class classifier_node():
        """
        A node definition for a NDT.
        """
        def __init__(self, model, clusters:List[List[int]], cluster_key:str, train: bool=True) -> None:
            """
            :param model: The classifier model the node uses. It's output should be equal to the number of clusters
            :param clusters: A list of clusters the node can classify an input into
            :param train: If True, the node can be trained. Otherwise, when fit is called, the node is not trained.
            """
            self.children = []
            self.classifier = model
            self.predict = self.classifier.predict
            self.clusters = clusters
            self.cluster_key = cluster_key
            self.train = train

    def __init__(
        self,
        classifier: Union["CLASSIFIER_TYPE", List["CLASSIFIER_TYPE"]],
        cluster_list: List[str],
        per_node_train: List[bool] = None
    ) -> None:
        """
        :param classifier: An instance of a `Classifier` who's is being used in the NDT or a list of classifiers to use
                           The interpretation of this list changes based on per_node_def
        :param cluster_list:  The clusters to group class labels by for each depth
        :param per_node_train: If True, `classifier` should be equal to the number of elements in cluster list. `classifier`
                             will be interpreted as if each entry is the classifier definition for that cluster only.
                             If False, `classifier` will be assumped to be composed of multiple `Classifier` instances
                             each with different output sizes. This is only checked if `classifier` is a list.
        """
        from art.estimators.classification import TensorFlowV2Classifier
        # Transform the classifier/classifier list into a dictionary for quick access
        classifier_dict = {}
        if isinstance(classifier, list):
            if len(classifier) == len(cluster_list):
                if per_node_train is not None and len(classifier) != len(per_node_train):
                    raise ValueError("When per_node_train is defined, the list of classifiers and \
                                      per_node_train must be the same length")
                for i, c in enumerate(classifier):
                    if isinstance(c, TensorFlowV2Classifier):
                        logger.info('NDT may not work properly with TF2 classifiers.')
                    classifier_dict[i] = c
            else:
                for c in classifier:
                    if c.nb_classes in classifier_dict.keys():
                        raise ValueError("Two classifiers defined for the same output shape. \
                                         Please only provide 1 classifier definition per output shape. \
                                         Note that if you are using a classifier where nb_class is not defined \
                                         it is automatically mapped to None.")
                    if isinstance(c, TensorFlowV2Classifier):
                        logger.info('NDT may not work properly with TF2 classifiers.')
                    classifier_dict[c.nb_classes] = c
        else:
            if isinstance(classifier, TensorFlowV2Classifier):
                logger.info('NDT may not work properly with TF2 classifiers.')
            classifier_dict[classifier.nb_classes] = classifier
           
        # Format the cluster_list
        if len(cluster_list) == 0:
            raise ValueError("cluster_list must have at least one class split defined")
        if not isinstance(cluster_list, list):
            raise ValueError("cluster_list should be a list of strings where classes in the same cluster are \
                             seperated by '-' and classes in different clusters are seperated by '_'")

        if per_node_train is not None:
            _per_node_train = {}
        else:
            _per_node_train = None

        processed_cluster_list = {}
        max_classes = 0

        # Transform the cluster list into a dictionary
        for c_ind, cluster in enumerate(cluster_list):
            cur_cluster, cluster_classes, cluster_key = self._process_cluster(cluster)
            processed_cluster_list[cluster_key] = cur_cluster

            if per_node_train is not None:
                _per_node_train[cluster_key] = per_node_train[c_ind]

            if len(cluster_classes) == 1:
                raise ValueError("There shouldn't be a cluster definition with only 1 value as it is redundant. \
                                  Please remove it from the list and use it's parent instead for classification")
            elif len(cluster_classes) > max_classes:
                root_key = cluster_key
                max_classes = len(cluster_classes)
                class_mapping = {k:v for (v,k) in enumerate(cluster_classes)}

        # Build the tree and get the root note
        root = self._build_tree(root_key, classifier_dict, processed_cluster_list, _per_node_train)

        super().__init__(model=None, clip_values=root.classifier.clip_values) # Just calling this to avoid errors
        self.classifier = root

        self._root_key = root_key
        self._class_mapping = class_mapping
        self.classifier_dict = classifier_dict
        self.cluster_list = processed_cluster_list
        self._per_node_train = _per_node_train
       
        # Get nb_classes by adding the leaf outputs
        self._nb_outputs = 0 # This is a count of the number of groups the NDT can predict
        self._nb_classes = len(class_mapping) # This is a count of the true number of classes

        queue = [root]
        while queue:
            node = queue.pop(0)
            for child_ind, child_node in enumerate(node.children):
                if child_node is not None:
                    queue.append(child_node)
                else:
                    self._nb_outputs += 1
                    
    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self.classifier.classifier._input_shape  # type: ignore

    # pylint: disable=W0221
    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> np.ndarray:
        """
        Perform prediction of the classifier for input `x`.

        :param x: Features in array of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param batch_size: Size of batches.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        input_inds = np.arange(len(x))
        predictions = np.zeros((len(x), self._nb_classes))
        queue = [[input_inds, self.classifier]]
        while queue:
            inds, node = queue.pop(0)
           
            node_predictions = np.argmax(node.classifier.predict(x[inds]), axis=1)
            for child_ind, child_node in enumerate(node.children):
                matching_inds = inds[node_predictions == child_ind]

                if child_node is not None:
                    queue.append([matching_inds, child_node])
                else:
                    cols = [self._class_mapping[v] for v in node.clusters[child_ind]] # This is for an edge case when the
                                                                                      # cluster list is not continuous
                    predictions[np.transpose([matching_inds]), np.array([cols])] = 1
        return predictions

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Fit the classifier using the training data `(x, y)`.

        :param x: Features in array of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param y: Target values (class labels in classification) in array of shape (nb_samples, nb_classes) in
                  one-hot encoding format.
        :param train_dict: A dictionary of framework-specific arguments for each classifier's fit function. 
                           If found, kwargs is ignored. Use if you need varied arguments for fit
        :param kwargs: Dictionary of framework-specific arguments.
        """

        y = check_and_transform_label_format(y, self.nb_classes)
        train_dict = kwargs.get("train_dict")
        if train_dict is not None:
            if not isinstance(train_dict, dict):
                raise ValueError("The argument train_dict needs to be a dictionary.")
            for key in list(train_dict.keys()):
                _,_, new_key = self._process_cluster(key)
                train_dict[new_key] = train_dict.pop(key)

        queue = [self.classifier]
        while queue:
            node = queue.pop(0)

            # First, populate the queue with the next training partners
            for child_node in node.children:
                if child_node is not None:
                    queue.append(child_node)

            # Next, adjust the data so the labels correspond to the difference children
            # TODO: This probably can be improved by passing the modified lists to the correct children
            modified_x = None
            modified_y = None
            for new_label, cluster in enumerate(node.clusters):
                ind = np.where([v in cluster for v in np.argmax(y, axis=1)])[0]

                new_y = np.zeros((len(y[ind]), len(node.clusters)))
                new_y[:,new_label] = 1
                if modified_x is None:
                    modified_x = x[ind]
                    modified_y = new_y
                else:
                    modified_x = np.append(modified_x, x[ind], axis=0)
                    modified_y = np.append(modified_y, new_y, axis=0)
           
            if node.train:
                try:
                    if train_dict is not None:
                        node.classifier.fit(modified_x, modified_y, **train_dict[node.cluster_key])
                    else:
                        node.classifier.fit(modified_x, modified_y, **kwargs) # Call the framework specific fit
                except NotImplementedError:
                    logger.warning('The classifier for the node with cluster', node.clusters, 'does not have a fit function  \
                                    so we are skipping it and moving on. If you need this node to be trained, please \
                                    train before initialization')
                    continue

    def save(self, filename: str, path: Optional[str] = None) -> None:
        """
        Save a model to file in the format specific to the backend framework. For Keras, .h5 format is used.

        :param filename: Name of the file where to store the model.
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `ART_DATA_PATH`.
        :raises `NotImplementedException`: This method is not supported for black-box classifiers.
        """
        raise NotImplementedError
        
    def _process_cluster(self, cluster: str) -> Union[List[List[int]], List[List[int]], str]:
        """
        This is a helper function. Given an '_' seperated cluster list, it will split the list into groups
        and return the clusters, the class labels in the cluster, and a string that can used to index the cluster
        """
        cur_cluster = []
        while cluster.find("_") != -1:
            cluster = cluster.split("_", 1)
            cur_cluster.append([int(i) for i in cluster[0].split("-")])
            cluster = cluster[1]
        cur_cluster.append([int(i) for i in cluster.split("-")])

        cluster_classes = sorted([int(i) for sublist in cur_cluster for i in sublist])
        cluster_key = '-'.join(map(str,cluster_classes))
        
        return cur_cluster, cluster_classes, cluster_key
        
    def _build_tree(self, cluster_key: str , classifier_dict: Dict[int, "CLASSIFIER_TYPE"],
                   cluster_dict: Dict[str, List[List[int]]], per_node_train: List[bool] = None) -> classifier_node:
        """
        This is a recursive helper function. The input is the key of the current node to build, the list of classifiers, the 
        dictionary of (cluster_key, class label) pairs, and a boolean to indicate which nodes should be trained. It returns the 
        root of the tree 
        """
        # First, we find the root. The root should be the cluster with the most classes.
        clusters = cluster_dict[cluster_key]
        if len(classifier_dict.keys()) == len(cluster_dict.keys()):
            cluster_ind = list(cluster_dict.keys()).index(cluster_key) # Dictionaries are insert ordered
            model = copy.deepcopy(classifier_dict[cluster_ind])
        else:
            if len(clusters) in classifier_dict.keys():
                model = copy.deepcopy(classifier_dict[len(clusters)])
            elif None in classifier_dict.keys(): #This can happen for some classifiers who decide the shape after training
                model = copy.deepcopy(classifier_dict[None])

        train = per_node_train[cluster_key] if per_node_train is not None else True
        cur_node = self.classifier_node(model, clusters, cluster_key, train)


        for subcluster in clusters:
            subcluster_key = '-'.join(sorted(map(str,subcluster)))
            if subcluster_key in cluster_dict.keys():
                cur_node.children.append(self._build_tree(subcluster_key, classifier_dict, cluster_dict, per_node_train))
            else:
                cur_node.children.append(None)

        return cur_node

# Optional Helper Function for generating a cluster list
def auto_cluster(model_weights: np.ndarray, nb_classes: int, split_num: Union[int,List[int]]) -> List[str]:
    """
    A helper function to automatically generate a cluster_list to use when creating a NDT.

    :param model_weights: The model weights. This should be in the shape (nb_classes, nb_weights). It will attempted to correct
                          The weight shape if the first dimension appears incorrect.
    :param nb_classes: The number of classes that need to be split.
    :param split_num: The number of splits that should be perfored at each level of the tree. If an int, the same split
                      number used everywhere. If a list, at each level, the split number at the from of the list is used.
                      If there are less elements in the list compared to the level, the last element in the list is used.
                      If the split number is greater than the number of remaining classes, the split number is changed to
                      be equal to the number of remaining classes.
    :return A list of strings, where each entry is defines a cluster and the approriate splits
    """
    logger.info("Make sure that the weights are passed with (nb_classes as the first dimension)")
    if np.shape(model_weights)[0]  != nb_classes:
        model_weights = np.transpose(model_weights)

    # Start of helper definition
    # Define a simple node for splitting
    class split_node():
        """
        A node definition used by auto_cluster
        """
        def __init__(self, classes:List[int], child_num: int=0) -> None:
            """
            :param classes: The numerical class indices that the node identifies
            :param child_num:  The numer of children the node has
            """
            self.classes = classes

            if child_num == 0:
                self.children = None
            else:
                self.children = [None for _ in range(child_num)]


    # Define a recurive function to build the tree for auto clustering
    def build_node(classes: List[int], weights: np.ndarray, split_num: Union[int, List[int]]) -> split_node:
        """
        This is used by auto_cluster to recursively build a decision tree based on the similarity of the weights passed.

        :param classes: The numerical class indicies
        :param weights: The model weights. This should be in the shape (nb_classes, nb_weights)
        :param split_num: The number of splits that should be perfored at each level of the tree. If an int, the same split
                          number used everywhere. If a list, at each level, the split number at the from of the list is used.
                          If there are less elements in the list compared to the level, the last element in the list is used.
                          If the split number is greater than the number of remaining classes, the split number is changed to
                          be equal to the number of remaining classes.
        return The root node of the tree
        """
        cur_node = split_node(classes)

        if len(classes) >= 2:
            if isinstance(split_num, list):
                cur_split_num = split_num.pop(0)
                if len(split_num) == 1:
                    split_num = split_num[0]
            else:
                cur_split_num = split_num

            if cur_split_num > len(classes):
                cur_split_num = len(classes)

            temp_weights = [weights[i] for i in range(len(weights)) if i in classes]
            model = AgglomerativeClustering(n_clusters=cur_split_num, affinity='euclidean', linkage='ward')
            model.fit(temp_weights)

            cur_node = split_node(classes, cur_split_num)
            labels = model.labels_

            for l in range(cur_split_num):
                if isinstance(split_num, list):
                    cur_node.children[l] = build_node([classes[i] for i in range(len(labels)) if labels[i] == l], weights, split_num.copy())
                else:
                    cur_node.children[l] = build_node([classes[i] for i in range(len(labels)) if labels[i] == l], weights, split_num)

        return cur_node
    # End of helper defintiions

    root = build_node(np.arange(nb_classes), model_weights, split_num)

    # Walk through the tree again to get the structure
    node_splits = []
    node_list = [root]
    while len(node_list) > 0:
        next_node = node_list.pop()
        cur_split = None
        if next_node.children is not None:
            for i in range(len(next_node.children)):
                if next_node.children[i] is not None:
                    node_list.append(next_node.children[i])
                    if cur_split is not None:
                        cur_split = cur_split + "_" + "-".join([str(l) for l in next_node.children[i].classes])
                    else:
                        cur_split = "-".join([str(l) for l in next_node.children[i].classes])

            node_splits.append(cur_split)
    logger.info("Automated Clusters:")
    logger.info(node_splits)
    return node_splits
    
