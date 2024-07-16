"""
Manages the tree representation for the APC model. It includes the ``TreeRepresentation`` class which handles the construction and manipulation of decision trees 
and ``TreeNode`` class that represents a node in the tree. 
This module is crucial for profiling aggregated data and extracting valuable insights
"""
import json
from typing import Union, Any

from pandas import DataFrame, Series
import numpy as np

from MED3pa.models.concrete_regressors import DecisionTreeRegressorModel
from .profiles import Profile

def to_serializable(obj: Any, additional_arg: Any = None) -> Any:
    """Convert an object to a JSON-serializable format.

    Args:
        obj (Any): The object to convert.

    Returns:
        Any: The JSON-serializable representation of the object.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, Profile):
        if additional_arg is not None:
            return obj.to_dict(additional_arg)
        else:
            return obj.to_dict()
    if isinstance(obj, _TreeNode):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj

class TreeRepresentation:
    """
    Represents the structure of a decision tree for a given set of features.
    """
    def __init__(self, features: list) -> None:
        """
        Initializes the TreeRepresentation with a list of feature names.

        Args:
            features (List[str]): List of feature names used in the decision tree.
        """
        self.features = features
        self.head = None
        self.nb_nodes = 0

    def build_tree_deprecated(self, dtr: DecisionTreeRegressorModel, X: DataFrame, y: Series, node_id: int = 0, path: list = ['*']) -> '_TreeNode':
        """
        Recursively builds the tree representation starting from the specified node.

        Args:
            dtr (DecisionTreeRegressorModel): Trained decision tree regressor model.
            X (DataFrame): Training data observations.
            y (Series): Training data labels.
            node_id (int): Node ID to start building from. Defaults to 0.
            path (Optional[List[str]]): Path to the current node. Defaults to ['*'].

        Returns:
            _TreeNode: The root node of the tree representation.
        """
        self.nb_nodes += 1
        left_child = dtr.model.tree_.children_left[node_id]
        right_child = dtr.model.tree_.children_right[node_id]

        node_value = y.mean()
        node_max = y.max()
        node_samples_ratio = dtr.model.tree_.n_node_samples[node_id] / dtr.model.tree_.n_node_samples[0] * 100

        # If we are at a leaf
        if left_child == -1:
            curr_node = _TreeNode(value=node_value, value_max=node_max, samples_ratio=node_samples_ratio,
                                  node_id=self.nb_nodes, path=path)
            return curr_node

        node_thresh = dtr.model.tree_.threshold[node_id]
        node_feature_id = dtr.model.tree_.feature[node_id]
        node_feature = self.features[node_feature_id]

        curr_path = list(path)  # Copy the current path to avoid modifying the original list
        curr_node = _TreeNode(value=node_value, value_max=node_max, samples_ratio=node_samples_ratio,
                              threshold=node_thresh, feature=node_feature, feature_id=node_feature_id,
                              node_id=self.nb_nodes, path=curr_path)

        # Update paths for child nodes
        left_path = curr_path + [f"{node_feature} <= {node_thresh}"]
        right_path = curr_path + [f"{node_feature} > {node_thresh}"]

        curr_node.c_left = self.build_tree(dtr, X=X.loc[X[node_feature] <= node_thresh], y=y[X[node_feature] <= node_thresh],
                                           node_id=left_child, path=left_path)
        curr_node.c_right = self.build_tree(dtr, X=X.loc[X[node_feature] > node_thresh], y=y[X[node_feature] > node_thresh],
                                            node_id=right_child, path=right_path)

        return curr_node

    def build_tree(self, dtr: DecisionTreeRegressorModel = None, X: DataFrame = None, y: Series = None, 
               node_id: int = 0, path: list = ['*'], loaded_tree: dict = None) -> '_TreeNode':
        """
        Recursively builds the tree representation starting from the specified node, either from a trained decision tree model
        or from a JSON structure.

        Args:
            dtr (DecisionTreeRegressorModel, optional): Trained decision tree regressor model.
            X (DataFrame, optional): Training data observations.
            y (Series, optional): Training data labels.
            node_id (int): Node ID to start building from. Defaults to 0.
            path (Optional[List[str]]): Path to the current node. Defaults to ['*'].
            json_node (dict, optional): JSON structure representing the tree node.

        Returns:
            _TreeNode: The root node of the tree representation.
        """
        self.nb_nodes += 1
        # Building tree from JSON structure
        if loaded_tree is not None:
            node_feature = loaded_tree.get("feature")
            node_thresh = loaded_tree.get("threshold")
            node_feature_id = loaded_tree.get("feature_id")

            if node_feature is not None and node_thresh is not None:
                left_mask = X[node_feature] <= node_thresh
                right_mask = X[node_feature] > node_thresh

                curr_node = _TreeNode(
                    value=y.mean(), value_max=y.max(), samples_ratio=len(y) / len(X) * 100,
                    threshold=node_thresh, feature=node_feature, feature_id=node_feature_id,
                    node_id=self.nb_nodes, path=loaded_tree.get("path", [])
                )

                if "c_left" in loaded_tree:
                    curr_node.c_left = self.build_tree(X=X[left_mask], y=y[left_mask], loaded_tree=loaded_tree["c_left"])
                if "c_right" in loaded_tree:
                    curr_node.c_right = self.build_tree(X=X[right_mask], y=y[right_mask], loaded_tree=loaded_tree["c_right"])
            else:
                curr_node = _TreeNode(
                    value=y.mean(), value_max=y.max(), samples_ratio=len(y) / len(X) * 100,
                    node_id=self.nb_nodes, path=loaded_tree.get("path", [])
                )
            return curr_node 
        else: 
            # Building tree from DecisionTreeRegressorModel
            left_child = dtr.model.tree_.children_left[node_id]
            right_child = dtr.model.tree_.children_right[node_id]

            node_value = y.mean()
            node_max = y.max()
            node_samples_ratio = dtr.model.tree_.n_node_samples[node_id] / dtr.model.tree_.n_node_samples[0] * 100

            # If we are at a leaf
            if left_child == -1:
                curr_node = _TreeNode(value=node_value, value_max=node_max, samples_ratio=node_samples_ratio,
                                    node_id=self.nb_nodes, path=path)
                return curr_node

            node_thresh = dtr.model.tree_.threshold[node_id]
            node_feature_id = dtr.model.tree_.feature[node_id]
            node_feature = self.features[node_feature_id]

            curr_path = list(path)  # Copy the current path to avoid modifying the original list
            curr_node = _TreeNode(value=node_value, value_max=node_max, samples_ratio=node_samples_ratio,
                                threshold=node_thresh, feature=node_feature, feature_id=node_feature_id,
                                node_id=self.nb_nodes, path=curr_path)

            # Update paths for child nodes
            left_path = curr_path + [f"{node_feature} <= {node_thresh}"]
            right_path = curr_path + [f"{node_feature} > {node_thresh}"]
            curr_node.c_left = self.build_tree(dtr, X=X.loc[X[node_feature] <= node_thresh], y=y[X[node_feature] <= node_thresh],
                                            node_id=left_child, path=left_path)
            curr_node.c_right = self.build_tree(dtr, X=X.loc[X[node_feature] > node_thresh], y=y[X[node_feature] > node_thresh],
                                                node_id=right_child, path=right_path)

            return curr_node

    def get_all_profiles(self, min_ca: float = 0, min_samples_ratio: float = 0) -> list:
        """
        Retrieves all profiles from the tree that meet the minimum criteria for value and sample ratio.

        Args:
            min_ca (float): Minimum value threshold for profiles. Defaults to 0.
            min_samples_ratio (float): Minimum sample ratio threshold for profiles. Defaults to 0.

        Returns:
            List[Profile]: A list of Profile instances meeting the specified criteria.
        """
        if self.head is None:
            raise ValueError("Tree has not been built yet.")
        profiles = self.head.get_profile(min_samples_ratio=min_samples_ratio, min_ca=min_ca)
        return profiles

    def get_all_nodes(self) -> list:
        """
        Retrieves all nodes from the tree with their paths.

        Returns:
            List[dict]: A list of dictionaries representing nodes with their paths.
        """
        if self.head is None:
            raise ValueError("Tree has not been built yet.")
        return self.head.get_all_nodes()
    
    def save_tree(self, file_path: str) -> None:
        """
        Saves the tree structure to a JSON file.

        Args:
            file_path (str): The file path where the tree structure will be saved.
        """
        if self.head is None:
            raise ValueError("Tree has not been built yet.")
        
        tree_dict = {}
        tree_dict = self.head.to_dict()
        tree_dict['features'] = self.features
        with open(file_path, 'w') as file:
            json.dump(tree_dict, file, default=to_serializable, indent=4)
    

class _TreeNode:
    """
    Represents a node in the tree structure.
    """
    def __init__(self, value: float =None, value_max: float=None, samples_ratio: float=None, threshold: float = None,
                 feature: str = None, feature_id: int = None, node_id: int = 0, path: list = None) -> None:
        """
        Initializes a _TreeNode object.

        Args:
            value (float): The average value at the node.
            value_max (float): The maximum value at the node.
            samples_ratio (float): The percentage of total samples present at the node.
            threshold (Optional[float]): The threshold used for splitting at this node. Defaults to None.
            feature (Optional[str]): The feature used for splitting at this node. Defaults to None.
            feature_id (Optional[int]): The identifier of the feature used for splitting. Defaults to None.
            node_id (int): Unique identifier for the node. Defaults to 0.
            path (Optional[List[str]]): The path from the root to this node. Defaults to an empty list.
        """
        self.c_left = None
        self.c_right = None
        self.value = value
        self.value_max = value_max
        self.samples_ratio = samples_ratio
        self.threshold = threshold
        self.feature = feature
        self.feature_id = feature_id
        self.node_id = node_id
        self.path = path if path is not None else []

    def assign_node(self, X: Union[DataFrame, Series]) -> float:
        """
        Assigns a value to a node based on input observations, navigating the tree until a leaf node is reached.

        Args:
            X (Union[DataFrame, Series]): Input observations used to navigate and determine the value at a node.
            min_samples_ratio (float): The minimum sample ratio to consider a node as valid for value assignment. Nodes
                                    with a sample ratio below this threshold will use the value from the nearest valid ancestor.
                                    Defaults to 0, which considers all nodes valid regardless of sample ratio.

        Returns:
            float: The value assigned based on the input observations and the structure of the tree.

        Raises:
            TypeError: If the input X is neither a pandas DataFrame nor a pandas Series.
        """
        # Check if the current node is a leaf node
        if self.c_left is None and self.c_right is None:
            return self.value

        if isinstance(X, DataFrame):
            X_value = X[self.feature].values[0]
        elif isinstance(X, Series):
            X_value = X[self.feature]
        else:
            raise TypeError(f"Parameter X is of type {type(X)}, but it must be of type 'pandas.DataFrame' or 'pandas.Series'.")

        if X_value <= self.threshold:  # If node split condition is true, then left children
            c_node = self.c_left
        else:
            c_node = self.c_right

        return c_node.assign_node(X)

    def assign_node_deprecated(self, X: Union[DataFrame, Series], depth: int = None, min_samples_ratio: float = 0) -> float:
        """
        Assigns a value to a node based on input observations, potentially navigating the tree up to a certain depth.

        Args:
            X (Union[DataFrame, Series]): Input observations used to navigate and determine the value at a node.
            depth (Optional[int]): The maximum depth to navigate in the tree for value assignment. Defaults to None,
                                   which means navigating until a leaf node is reached.
            min_samples_ratio (float): The minimum sample ratio to consider a node as valid for value assignment. Nodes
                                       with a sample ratio below this threshold will use the value from the nearest valid ancestor.
                                       Defaults to 0, which considers all nodes valid regardless of sample ratio.

        Returns:
            float: The value assigned based on the input observations and the structure of the tree.

        Raises:
            TypeError: If the input X is neither a pandas DataFrame nor a pandas Series.
        """
        if depth == 0 or self.c_left is None:
            return self.value

        if isinstance(X, DataFrame):
            X_value = X[self.feature].values[0]
        elif isinstance(X, Series):
            X_value = X[self.feature]
        else:
            raise TypeError(f"Parameter X is of type {type(X)}, but it must be of type 'pandas.DataFrame' or 'pandas.Series'.")

        if depth is not None:
            depth -= 1

        if X_value <= self.threshold:  # If node split condition is true, then left children
            c_node = self.c_left
        else:
            c_node = self.c_right
        
        if c_node.samples_ratio < min_samples_ratio:  # If not enough samples in child node
            return self.value
        
        return c_node.assign_node(X, depth, min_samples_ratio)
    
    def get_profile(self, min_samples_ratio: float, min_ca: float) -> list:
        """
        Retrieves profiles from the subtree rooted at this node that meet the specified criteria.

        Args:
            min_samples_ratio (float): The minimum sample ratio a node must have to be included in the output profiles.
            min_ca (float): The minimum value a node must have to be included in the output profiles.

        Returns:
            List[Profile]: A list of Profile instances representing nodes that meet the criteria.
        
        """
        profiles = []
        if self.c_left is not None and self.c_left.samples_ratio >= min_samples_ratio:
            # Recursively retrieve profiles from the left child
            profiles.extend(self.c_left.get_profile(min_samples_ratio, min_ca))

        if self.c_right is not None and self.c_right.samples_ratio >= min_samples_ratio:
            # Recursively retrieve profiles from the right child
            profiles.extend(self.c_right.get_profile(min_samples_ratio, min_ca))

        # Check if the current node meets the criteria
        if self.samples_ratio >= min_samples_ratio and self.value >= min_ca:
            profile = Profile(node_id=self.node_id, path=self.path)
            profiles.append(profile)

        return profiles

    def get_all_nodes(self) -> list:
        """
        Retrieves all nodes in the subtree rooted at this node with their paths.

        Returns:
            List[dict]: A list of dictionaries representing nodes with their paths.
        """
        nodes = [{
            'node_id': self.node_id,
            'path': self.path
        }]
        
        if self.c_left is not None:
            nodes.extend(self.c_left.get_all_nodes())
        
        if self.c_right is not None:
            nodes.extend(self.c_right.get_all_nodes())
        
        return nodes
    
    def to_dict(self) -> dict:
        """
        Converts the node and its children to a dictionary.

        Returns:
            dict: A dictionary representation of the node and its children.
        """
        node_dict = {
            'threshold': self.threshold,
            'feature': self.feature,
            'feature_id': self.feature_id,
            'node_id': self.node_id,
            'path': self.path
        }
        if self.c_left is not None:
            node_dict['c_left'] = self.c_left.to_dict()
        if self.c_right is not None:
            node_dict['c_right'] = self.c_right.to_dict()
        return node_dict
    
