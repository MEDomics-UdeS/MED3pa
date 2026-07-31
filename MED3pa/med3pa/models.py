"""Defines the models used within the MED3pa framework. It includes classes for Individualized Predictive Confidence
(IPC) models that predict uncertainty at an individual level, where the regressor type can be specified by the user.
Additionally, it includes Aggregated Predictive Confidence (APC) models that predict uncertainty for groups of
similar data points, and Mixed Predictive Confidence (MPC) models that combine the predictions from IPC and APC
models.
"""

import json
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from typing import Any, Dict, List, Optional

from MED3pa.med3pa.tree import TreeRepresentation
from MED3pa.models.concrete_regressors import (DecisionTreeRegressorModel, RandomForestRegressorModel,
                                               EnsembleRandomForestRegressorModel)
from MED3pa.models.data_strategies import ToDataframesStrategy
from MED3pa.models.regression_metrics import RegressionEvaluationMetrics
from MED3pa.models import rfr_params, dtr_params
from abc import ABC, abstractmethod

class AbstractUncertaintyEstimator:
    default_params = {}  # {'random_state': 54288}

    supported_regressors_mapping = {
        'RandomForestRegressor': RandomForestRegressorModel,
        'EnsembleRandomForestRegressor': EnsembleRandomForestRegressorModel,
        'DecisionTreeRegressor': DecisionTreeRegressorModel
    }

    supported_regressors_params = {
        'RandomForestRegressor': {
            'params': rfr_params.rfr_params,
            'grid_params': rfr_params.rfr_gridsearch_params
        },
        'EnsembleRandomForestRegressor': {
            'params': rfr_params.rfr_params,
            'grid_params': rfr_params.rfr_gridsearch_params
        }
    }

    def __init__(self, model_name: str,
                 params: Optional[Dict[str, Any]] = None,
                 pretrained_model: Optional[str] = None,
                 random_state: Optional[int] = None):
        """
        Initializes the AbstractUncertaintyEstimator class instance.

        Args:
            model_name (str): Name of the model.
            params (Optional[Dict[str, Any]]): Parameters to initialize the regression model, default is None.
            pretrained_model (Optional[str]): Path to a pretrained model, default is None.
        """
        if model_name not in self.supported_regressors_mapping:
            raise ValueError(
                f"Unsupported model name: {model_name}. Supported models are: "
                f"{list(self.supported_regressors_mapping)}")

        model_class = self.supported_regressors_mapping[model_name]

        if params is None:
            params = self.default_params.copy()
        if 'random_state' not in params:
            params['random_state'] = random_state

        self.model = model_class(params)
        self.params = params
        self.grid_search_params = {}
        self.optimized = False
        self.pretrained = False
        self.model_name = model_name

        if pretrained_model is not None:
            self.load_model(pretrained_model)

    def evaluate(self, X: np.ndarray, y: np.ndarray, eval_metrics: List[str], print_results: bool = False
                 ) -> Dict[str, float]:
        """
        Evaluates the model using specified metrics.

        Args:
            X (np.ndarray): observations for evaluation.
            y (np.ndarray): True labels for evaluation.
            eval_metrics (List[str]): Metrics to use for evaluation.
            print_results (bool): Whether to print the evaluation results.

        Returns:
            Dict[str, float]: A dictionary with metric names and their evaluated scores.
        """
        evaluation_results = self.model.evaluate(X, y, eval_metrics, print_results)
        return evaluation_results

    def get_info(self) -> Dict[str, Any]:
        """
        Returns information about the AbstractUncertaintyEstimator instance.

        Returns:
            Dict[str, Any]: A dictionary containing the model name, parameters, whether the model was optimized, and other relevant details.
        """
        return {
            'model_name': self.model_name,
            'params': self.params if not self.pretrained else {},
            'optimized': self.optimized,
            'grid_search_params': self.grid_search_params,
            'pretrained': self.pretrained
        }

    def save_model(self, file_path: str) -> None:
        """
        Saves the trained model to a pickle file.

        Args:
            file_path (str): The path to the file where the model will be saved.
        """
        model_type = "IPCModel" if isinstance(self, IPCModel) else "APCModel"

        with open(f"{file_path}_{model_type}.pkl", 'wb') as file:
            pickle.dump(self.model, file)

    def load_model(self, file_path: str) -> None:
        """
        Loads a pre-trained model from a pickle file.

        Args:
            file_path (str): The path to the pickle file.
        """
        with open(file_path, 'rb') as file:
            loaded_model = pickle.load(file)

        if not isinstance(loaded_model, self.supported_regressors_mapping[self.model_name]):
            raise TypeError(f"The loaded model type does not match the specified model type: {self.model_name}")

        self.model = loaded_model
        self.pretrained = True


class IPCModel(AbstractUncertaintyEstimator):
    """
    IPCModel class used to predict the Individualized predicted confidence. ie, the base model confidence for each data
    point.
    """
    default_params = {}  # {'random_state': 54288}

    def __init__(self, model_name: str = 'RandomForestRegressor', params: Optional[Dict[str, Any]] = None,
                 pretrained_model: Optional[str] = None, random_state: Optional[int] = None) -> None:
        """
        Initializes the IPCModel with a regression model class name and optional parameters.

        Args:
            model_name (str): The name of the regression model class to use, default is 'RandomForestRegressor'.
                Allowed values are in IPCModel.supported_regressors_mapping.
            params (Optional[Dict[str, Any]]): Parameters to initialize the regression model, default is None.
            pretrained_model (Optional[str]): Path to a pretrained regression model, serving as ipc model,
                default is None.
            random_state (Optional[int]): Random state to apply to the model initialization.
        """
        super().__init__(model_name=model_name, params=params, pretrained_model=pretrained_model,
                         random_state=random_state)

    @classmethod
    def supported_ipc_models(cls) -> List:
        """
        Returns a list of supported IPC models.

        Returns:
            list: A list of supported regression model names.
        """
        return list(AbstractUncertaintyEstimator.supported_regressors_mapping)

    @classmethod
    def supported_models_params(cls) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dictionary containing the supported models and their parameters and grid search parameters.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary with model names as keys and another dictionary as value containing 
                                    'params' and 'grid_search_params' for each model.
        """
        return AbstractUncertaintyEstimator.supported_regressors_params

    def optimize(self, param_grid: dict, cv: int, x: np.ndarray, confidence_score: np.ndarray,
                 sample_weight: np.ndarray = None) -> None:
        """
        Optimizes the model parameters using GridSearchCV.

        Args:
            param_grid (Dict[str, Any]): The parameter grid to explore.
            cv (int): The number of cross-validation folds.
            confidence_score (np.ndarray): The confidence scores to train the IPCModel.
            x (np.ndarray): Training data.
            sample_weight (Optional[np.ndarray]): Weights for the training samples.
        """
        if sample_weight is None:
            sample_weight = np.full(x.shape[0], 1)

        grid_search = GridSearchCV(estimator=self.model.model, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=0)
        grid_search.fit(x, confidence_score, sample_weight=sample_weight)

        self.model.set_model(grid_search.best_estimator_)
        self.model.update_params(grid_search.best_params_)
        self.params.update(grid_search.best_params_)
        self.grid_search_params = param_grid
        self.optimized = True

    def train(self, x: np.ndarray, confidence_score: np.ndarray, **params) -> None:
        """
        Trains the model on the provided training data and error probabilities.

        Args:
            x (np.ndarray): Feature matrix for training.
            confidence_score (np.ndarray): The confidence scores corresponding to each training instance.
        """
        self.model.train(x, confidence_score, **params)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts error probabilities for the given input observations using the trained model.

        Args:
            x (np.ndarray): Feature matrix for which to predict error probabilities.

        Returns:
            np.ndarray: Predicted error probabilities.
        """
        return self.model.predict(x)


class APCModel(AbstractUncertaintyEstimator):
    """
    APCModel class used to predict the Aggregated predicted confidence. ie, the base model confidence for a group of
    similar data points.
    """
    default_params = {'max_depth': 3, 'min_samples_leaf': 1, 'random_state': 54288}

    supported_params = {
        'DecisionTreeRegressor': {
            'params': dtr_params.dtr_params,
            'grid_params': dtr_params.dtr_gridsearch_params
        }
    }

    def __init__(self, features: List[str], params: Optional[Dict[str, Any]] = None,
                 tree_file_path: Optional[str] = None, pretrained_model: Optional[str] = None,
                 model_name: str = "DecisionTreeRegressor", random_state: Optional[int] = None) -> None:
        """
        Initializes the APCModel with the necessary components to perform tree-based regression and to build a tree
        representation.

        Args:
            features (List[str]): List of features used in the model.
            params (Optional[Dict[str, Any]]): Parameters to initialize the regression model, default is settings for
                a basic decision tree.
            tree_file_path (Optional[str]): Path to the saved tree JSON file, default is None.
            pretrained_model (Optional[str]): Path to a pretrained DecisionTree model, serving as apc model,
                default is None.
            model_name (str): Name of the model, default is "DecisionTreeRegressor".
        """
        super().__init__(model_name=model_name, params=params, pretrained_model=pretrained_model,
                         random_state=random_state)

        self.treeRepresentation = TreeRepresentation(features=features)
        self.dataPreparationStrategy = ToDataframesStrategy()
        self.features = features
        self.loaded_tree = None

        if tree_file_path:
            self.load_tree(tree_file_path)

    def load_tree(self, file_path: str) -> None:
        """
        Loads the tree structure from a JSON file and initializes the tree representation.

        Args:
            file_path (str): The file path from which the tree structure will be loaded.
        """
        with open(file_path, 'r') as file:
            tree_dict = json.load(file)

        self.loaded_tree = tree_dict

    @classmethod
    def supported_models_params(cls) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dictionary containing the supported models and their parameters and grid search parameters.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary with model names as keys and another dictionary as value containing 
                                    'params' and 'grid_search_params' for each model.
        """
        return cls.supported_params

    def train(self, x: np.ndarray, error_prob: np.ndarray | pd.Series) -> None:
        """
        Trains the model using the provided data and error probabilities and builds the tree representation.

        Args:
            x (np.ndarray): Feature matrix for training.
            error_prob (np.ndarray | pd.Series): Error probabilities corresponding to each training instance.
        """
        if not self.pretrained:
            self.model.train(x, error_prob)
        df_X, df_y, df_w = self.dataPreparationStrategy.execute(column_labels=self.features, observations=x,
                                                                labels=error_prob)
        self.treeRepresentation.build_tree(self.model, df_X, error_prob)

    def optimize(self, param_grid: dict, cv: int, x: np.ndarray, confidence_score: np.ndarray,
                 sample_weight: np.ndarray = None) -> None:
        """
        Optimizes the model parameters using GridSearchCV.

        Args:
            param_grid (Dict[str, Any]): The parameter grid to explore.
            cv (int): The number of cross-validation folds.
            x (np.ndarray): Training data.
            confidence_score (np.ndarray): The confidence scores to train the APCModel.
            sample_weight (Optional[np.ndarray]): Weights for the training samples.
        """
        if sample_weight is None:
            sample_weight = np.full(x.shape[0], 1)
        grid_search = GridSearchCV(estimator=self.model.model, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=0)
        grid_search.fit(x, confidence_score, sample_weight=sample_weight)
        self.model.set_model(grid_search.best_estimator_)
        self.model.update_params(grid_search.best_params_)
        self.params.update(grid_search.best_params_)
        self.grid_search_params = param_grid
        df_X, df_y, df_w = self.dataPreparationStrategy.execute(column_labels=self.features, observations=x,
                                                                labels=confidence_score)
        self.treeRepresentation.build_tree(self.model, df_X, confidence_score)
        self.optimized = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts error probabilities using the tree representation for the given input observations.

        Args:
            X (np.ndarray): Feature matrix for which to predict error probabilities.

        Returns:
            np.ndarray: Predicted error probabilities based on the aggregated confidence levels.
        """
        if self.treeRepresentation.head is None:
            raise ValueError("The Tree Representation has not been initialized, try fitting the APCModel first.")

        df_X, _, _ = self.dataPreparationStrategy.execute(column_labels=self.features, observations=X, labels=None)
        predictions = []

        for index, row in df_X.iterrows():
            prediction = self.treeRepresentation.head.assign_node(row)
            predictions.append(prediction)

        return np.array(predictions)

class MpcStrategy(ABC):
    """Combines IPC and APC confidences into the mixed predicted confidence."""

    @abstractmethod
    def combine(self, ipc_values: np.ndarray, apc_values: np.ndarray) -> np.ndarray:
        """Return one mixed confidence per observation."""

    def name(self) -> str:
        """Short label for get_info() and saved configs."""
        return type(self).__name__
    
class MinimumStrategy(MpcStrategy):
    def combine(self, ipc_values, apc_values):
        return np.minimum(ipc_values, apc_values)
    def name(self):
        return "minimum"


class AverageStrategy(MpcStrategy):
    def combine(self, ipc_values, apc_values):
        return (ipc_values + apc_values) / 2
    def name(self):
        return "average"
    
class MPCModel:
    """
    MPCModel class used to predict the Mixed predicted confidence. ie, the compromise between the APC and IPC values.
    """
    strategy_mapping = {
            'minimum': MinimumStrategy,
            'average': AverageStrategy,
        }

    def __init__(self, IPC_model, APC_model,
                strategy: str | MpcStrategy = "minimum") -> None:
        self.IPC_model = IPC_model
        self.APC_model = APC_model
        if isinstance(strategy, MpcStrategy):
            self.strategy = strategy
        elif isinstance(strategy, str) and strategy in self.strategy_mapping:
            self.strategy = self.strategy_mapping[strategy]()      
        else:
            raise ValueError(
                f"Unrecognized MPC strategy. Available: {list(self.strategy_mapping)}"
            )

    @classmethod
    def supported_strategies(cls) -> list:
        return list(cls.strategy_mapping)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.strategy.combine(self.IPC_model.predict(X),
                                    self.APC_model.predict(X))

    def get_info(self) -> Dict[str, Any]:
        """
        Returns information about the MPC model.

        Returns:
            Dict[str, Any]: A dictionary containing the model name, parameters, whether the model was optimized,
            and other relevant details for both IPC and APC models.
        """
        ipc_infos = self.IPC_model.get_info()
        apc_infos = self.APC_model.get_info()
        return {
            'ipc_infos': ipc_infos,
            'apc_infos': apc_infos,
            'mpc_strategy': self.strategy.name()
        }

    def evaluate(self, X: np.ndarray, y: np.ndarray, eval_metrics: List[str], print_results: bool = False
                 ) -> Dict[str, float]:
        """
        Evaluates the model using specified metrics.

        Args:
            X (np.ndarray): observations for evaluation.
            y (np.ndarray): True labels for evaluation.
            eval_metrics (List[str]): Metrics to use for evaluation.
            print_results (bool): Whether to print the evaluation results.

        Returns:
            Dict[str, float]: A dictionary with metric names and their evaluated scores.
        """
        predictions = self.predict(X)
        evaluation_results = {}

        for metric_name in eval_metrics:
            metric_function = RegressionEvaluationMetrics.get_metric(metric_name)
            if metric_function:
                evaluation_results[metric_name] = metric_function(y, predictions)
            else:
                print(f"Error: The metric '{metric_name}' is not supported.")

        if print_results:
            print("Evaluation Results:")
            for metric, value in evaluation_results.items():
                print(f"{metric}: {value:.2f}")

        return evaluation_results

    def save_model(self, file_path: str) -> None:
        """
        Saves the trained model to a pickle file.

        Args:
            file_path (str): The path to the file where the model will be saved.
        """
        self.IPC_model.save_model(file_path=file_path)
        self.APC_model.save_model(file_path=file_path)
