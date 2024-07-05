"""
Defines the models used within the med3pa framework. It includes classes for Individualized Predictive Confidence (IPC) models that predict uncertainty at an individual level, 
where the regressor type can be specified by the user. 
Additionally, it includes Aggregated Predictive Confidence (APC) models that predict uncertainty for groups of similar data points, 
and Mixed Predictive Confidence (MPC) models that combine the predictions from IPC and APC models.
"""
from typing import Any, Dict, List, Optional, Type

import numpy as np
from sklearn.model_selection import GridSearchCV

from MED3pa.med3pa.tree import TreeRepresentation
from MED3pa.models.abstract_models import RegressionModel
from MED3pa.models.concrete_regressors import DecisionTreeRegressorModel, RandomForestRegressorModel
from MED3pa.models.data_strategies import ToDataframesStrategy
from MED3pa.models import rfr_params, dtr_params

class IPCModel:
    """
    IPCModel class used to predict the Individualized predicted confidence. ie, the base model confidence for each data point.
    """
    default_params = {'random_state': 54288}
    
    supported_regressors_mapping = {
        'RandomForestRegressor' : RandomForestRegressorModel
    }

    supported_regressos_params = {
            'RandomForestRegressor' : {
                'params' : rfr_params.rfr_params,
                'grid_params' : rfr_params.rfr_gridsearch_params
            }
    }
    
    def __init__(self, model_name: str = 'RandomForestRegressor', params: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the IPCModel with a regression model class name and optional parameters.

        Args:
            model_name (str): The name of the regression model class to use, default is 'RandomForestRegressor'.
            params (Optional[Dict[str, Any]]): Parameters to initialize the regression model, default is None.
        """
        if model_name not in self.supported_regressors_mapping:
            raise ValueError(f"Unsupported model name: {model_name}. Supported models are: {self.supported_ipc_models()}")

        model_class = self.supported_regressors_mapping[model_name]

        if params is None:
            params = self.default_params.copy()
        else:
            random_state_params = {'random_state': 54288}
            params.update(random_state_params)
        
        self.model = model_class(params)
        self.params = params
        self.optimized = False
        self.model_name = model_name
    
    @classmethod
    def supported_ipc_models(cls) -> list:
        """
        Returns a list of supported IPC models.

        Returns:
            list: A list of supported regression model names.
        """
        return list(cls.supported_regressors_mapping.keys())
    
    @classmethod
    def supported_models_params(cls) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dictionary containing the supported models and their parameters and grid search parameters.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary with model names as keys and another dictionary as value containing 
                                    'params' and 'grid_search_params' for each model.
        """
        return cls.supported_regressos_params
    
    def optimize(self, param_grid: dict, cv: int, x: np.ndarray, error_prob: np.ndarray, sample_weight: np.ndarray = None) -> None:
        """
        Optimizes the model parameters using GridSearchCV.

        Args:
            param_grid (Dict[str, Any]): The parameter grid to explore.
            cv (int): The number of cross-validation folds.
            x (np.ndarray): Training data.
            y (np.ndarray): Target data.
            sample_weight (Optional[np.ndarray]): Weights for the training samples.
        """
        if sample_weight is None:
            sample_weight = np.full(x.shape[0], 1)
        grid_search = GridSearchCV(estimator=self.model.model, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=0)
        grid_search.fit(x, error_prob, sample_weight=sample_weight)
        self.model.set_model(grid_search.best_estimator_)
        self.model.update_params(grid_search.best_params_)
        self.params.update(grid_search.best_params_)
        self.optimized = True

    def train(self, x: np.ndarray, error_prob: np.ndarray) -> None:
        """
        Trains the model on the provided training data and error probabilities.

        Args:
            x (np.ndarray): Feature matrix for training.
            error_prob (np.ndarray): Error probabilities corresponding to each training instance.
        """
        self.model.train(x, error_prob)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts error probabilities for the given input observations using the trained model.

        Args:
            x (np.ndarray): Feature matrix for which to predict error probabilities.

        Returns:
            np.ndarray: Predicted error probabilities.
        """
        return self.model.predict(x)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, eval_metrics: List[str], print_results: bool = False) -> Dict[str, float]:
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
        Returns information about the IPCModel instance.

        Returns:
            Dict[str, Any]: A dictionary containing the model name, parameters, whether the model was optimized, and other relevant details.
        """
        return {
            'model_name': self.model_name,
            'params': self.params,
            'optimized': self.optimized,
        }
    
class APCModel:
    """
    APCModel class used to predict the Aggregated predicted confidence. ie, the base model confidence for a group of similar data points.
    """
    default_params = {'max_depth': 3, 'min_samples_leaf': 1, 'random_state': 54288}
    
    supported_params = {
            'DecisionTreeRegressor' : {
                'params' : dtr_params.dtr_params,
                'grid_params' : dtr_params.dtr_gridsearch_params
            }
    }

    def __init__(self, features: list, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the APCModel with the necessary components to perform tree-based regression and to build a tree representation.

        Args:
            features (List[str]): List of features used in the model.
            model_class (Type[RegressionModel]): The regression model class to use for tree-based modeling, default is DecisionTreeRegressorModel.
            params (Optional[Dict[str, Any]]): Parameters to initialize the regression model, default is settings for a basic decision tree.
        """
        if params is None:
            params = self.default_params
        else:
            random_state_params = {'random_state': 54288}
            params.update(random_state_params)
            
        self.model = DecisionTreeRegressorModel(params)
        self.treeRepresentation = TreeRepresentation(features=features)
        self.dataPreparationStrategy = ToDataframesStrategy()
        self.features = features
        self.params = params
        self.optimized = False


    @classmethod
    def supported_models_params(cls) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dictionary containing the supported models and their parameters and grid search parameters.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary with model names as keys and another dictionary as value containing 
                                    'params' and 'grid_search_params' for each model.
        """
        return cls.supported_params
    
    def train(self, x: np.ndarray, error_prob: np.ndarray) -> None:
        """
        Trains the model using the provided data and error probabilities and builds the tree representation.

        Args:
            x (np.ndarray): Feature matrix for training.
            error_prob (np.ndarray): Error probabilities corresponding to each training instance.
        """
        self.model.train(x, error_prob)
        df_X, df_y, df_w = self.dataPreparationStrategy.execute(column_labels=self.features, observations=x, labels=error_prob)
        self.treeRepresentation.head = self.treeRepresentation.build_tree(self.model, df_X, error_prob, 0)
    
    def optimize(self, param_grid: dict, cv: int, x: np.ndarray, error_prob: np.ndarray, sample_weight: np.ndarray = None) -> None:
        """
        Optimizes the model parameters using GridSearchCV.

        Args:
            param_grid (Dict[str, Any]): The parameter grid to explore.
            cv (int): The number of cross-validation folds.
            x (np.ndarray): Training data.
            y (np.ndarray): Target data.
            sample_weight (Optional[np.ndarray]): Weights for the training samples.
        """
        if sample_weight is None:
            sample_weight = np.full(x.shape[0], 1)
        grid_search = GridSearchCV(estimator=self.model.model, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=0)
        grid_search.fit(x, error_prob, sample_weight=sample_weight)
        self.model.set_model(grid_search.best_estimator_)
        self.model.update_params(grid_search.best_params_)
        self.params.update(grid_search.best_params_)
        self.optimized = True
        

    def predict(self, X: np.ndarray, depth: int = None, min_samples_ratio: float = 0) -> np.ndarray:
        """
        Predicts error probabilities using the tree representation for the given input observations.

        Args:
            x (np.ndarray): Feature matrix for which to predict error probabilities.
            depth (Optional[int]): The maximum depth of the tree to use for predictions.

        Returns:
            np.ndarray: Predicted error probabilities based on the aggregated confidence levels.
        """
        df_X, _, _ = self.dataPreparationStrategy.execute(column_labels=self.features, observations=X, labels=None)
        predictions = []

        for index, row in df_X.iterrows():
            if self.treeRepresentation.head is not None:
                prediction = self.treeRepresentation.head.assign_node(row, depth, min_samples_ratio)
                predictions.append(prediction)
            else:
                raise ValueError("The Tree Representation has not been initialized, try fitting the APCModel first.")

        return np.array(predictions)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, eval_metrics: List[str], print_results: bool = False) -> Dict[str, float]:
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
        Returns information about the APCModel instance.

        Returns:
            Dict[str, Any]: A dictionary containing the model name, parameters, whether the model was optimized, and other relevant details.
        """
        return {
            'model_name': "DecisionTreeRegressor",
            'params': self.params,
            'optimized': self.optimized,
        }
    
class MPCModel:
    """
    MPCModel class used to predict the Mixed predicted confidence. ie, the minimum between the APC and IPC values.
    """
    def __init__(self, IPC_values: np.ndarray=None, APC_values: np.ndarray=None) -> None:
        """
        Initializes the MPCModel with IPC and APC values.

        Args:
            IPC_values (np.ndarray): IPC values.
            APC_values (np.ndarray): APC values.
        """
        self.IPC_values = IPC_values
        self.APC_values = APC_values

    def predict(self) -> np.ndarray:
        """
        Combines IPC and APC values to predict MPC values.

        Returns:
            np.ndarray: Combined MPC values.
        """
        if self.APC_values is None and self.IPC_values is None:
            raise ValueError("Both APC values and IPC values are not set!")
        
        if self.APC_values is None:
            MPC_values = self.IPC_values
        elif self.IPC_values is None:
            MPC_values = self.APC_values
        else:
            MPC_values = np.minimum(self.IPC_values, self.APC_values)

        return MPC_values
