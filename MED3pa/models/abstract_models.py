"""
The abstract_models.py module defines core abstract classes that serve as the foundation for model management in the system. 
It includes ``Model``, which standardizes basic operations like evaluation and parameter validation..etc across all models. 
It also introduces specialized abstract classes such as ``ClassificationModel`` and ``RegressionModel``, 
each adapting these operations to specific needs of classification and regression tasks.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import os
import pickle
import json

import numpy as np

from .data_strategies import DataPreparingStrategy

class Model(ABC):
    """
    An abstract base class for all models, defining a common API for model operations such as evaluation and parameter validation.

    Attributes:
        model (Any): The underlying model instance.
        model_class (type): The class type of the underlying model instance.
        params (dict): The params used for initializinf the model.
        data_preparation_strategy (DataPreparingStrategy): Strategy for preparing data before training or evaluation.
        pickled_model (Boolean): A boolean indicating whether or not the model has been loaded from a pickled file.
    """
    def __init__(self) -> None:
        super().__init__()
        self.model = None
        self.model_class = None
        self.params = None
        self.data_preparation_strategy = None
        self.pickled_model = False
        self.file_path = None

    def get_model(self) -> Any:
        """
        Retrieves the underlying model instance, which is typically a machine learning model object.

        Returns:
            Any: The underlying model instance if set, None otherwise.
        """
        return self.model
    
    def get_path(self) -> Any:
        """
        Retrieves the file path of the model if it has been loaded from a pickled file.

        Returns:
            str: The file path of the model if it has been loaded from a pickled file, None otherwise.
        """
        return self.file_path
    
    def get_model_type(self)-> Optional[str]:
        """
        Retrieves the class type of the underlying model instance, which indicates the specific 
        implementation of the model used.

        Returns:
            Optional[str]: The class of the model if set, None otherwise.
        """
        return self.model_class.__name__ if self.model_class else None
    
    def get_data_strategy(self) -> Optional[str]:
        """
        Retrieves the data preparation strategy associated with the model. This strategy handles 
        how data should be formatted before being passed to the model for training or evaluation.

        Returns:
            Optional[str]: The name of the current data preparation strategy if set, None otherwise.
        """
        return self.data_preparation_strategy.__class__.__name__ if self.data_preparation_strategy else None

    def get_params(self):
        """
        Retrieves the underlying model's parameters.

        Returns:
            Dict[str, Any]: the model's parameters.
        """
        return self.params
    
    def is_pickled(self):
        """
        Returns whether or not the model has been loaded from a pickled file.
        
        Returns:
            Boolean: has the model been loaded from a pickled file.
        """
        return self.pickled_model
    
    def set_model(self, model: Any) -> None:
        """
        Sets the underlying model instance and updates the model class to match the type of the given model.

        Args:
            model (Any): The model instance to be set.
        """
        self.model = model
        self.model_class = type(model)
    
    def set_params(self, params : dict):
        """
        Sets the parameters for the model. These parameters are typically used for model initialization or configuration.

        Args:
            params (Dict[str, Any]): A dictionary of parameters for the model.
        """
        self.params = params
    
    def set_file_path(self, file_path : str):
        """
        Sets the file path of the model. 

        Args:
            file_path (str): the file path of the model.
        """
        self.file_path = file_path

    def update_params(self, params : dict):
        """
        Updates the current model parameters by merging new parameter values from the given dictionary.
        This method allows for dynamic adjustment of model configuration during runtime.

        Args:
            params (Dict[str, Any]): A dictionary containing parameter names and values to be updated.
        """
        self.params.update(params)

    def set_data_strategy(self, strategy: DataPreparingStrategy):
        """
        Sets the underlying model's data preparation strategy.

        Args:
            strategy (DataPreparingStrategy): strategy to be used to prepare the data for training, validation...etc.
        """
        self.data_preparation_strategy = strategy
    
    def get_info(self) -> Dict[str, Any]:
        """
        Retrieves detailed information about the model.

        Returns:
            Dict[str, Any]: A dictionary containing information about the model's type, parameters, 
                            data preparation strategy, and whether it's a pickled model.
        """
        return {
            "model": self.__class__.__name__,
            "model_type": self.get_model_type(),
            "params": self.get_params(),
            "data_preparation_strategy": self.get_data_strategy() if self.get_data_strategy() else None,
            "pickled_model": self.is_pickled(),
            "file_path": self.get_path()
        }
    
    def save(self, path: str) -> None:
        """
        Saves the model instance as a pickled file and the parameters as a JSON file within the specified directory.
        
        Args:
            path (str): The directory path where the model and parameters will be saved.
        """
        # Create the directory if it does not exist
        if not os.path.exists(path):
            os.makedirs(path)

        # Define file paths
        model_path = os.path.join(path, 'model_instance.pkl')
        params_path = os.path.join(path, 'model_info.json')

        # Save the model as a pickled file
        with open(model_path, 'wb') as model_file:
            pickle.dump(self.model, model_file)

        # Save the parameters as a JSON file
        with open(params_path, 'w') as params_file:
            json.dump(self.get_info(), params_file)

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray, eval_metrics: List[str], print_results: bool = False) -> Dict[str, float]:
        """
        Evaluates the model using specified metrics.

        Args:
            X (np.ndarray): observations for evaluation.
            y (np.ndarray): True labels for evaluation.
            eval_metrics (List[str]): Metrics to use for evaluation.
            print_results (bool, optional): Whether to print the evaluation results. Defaults to False.

        Returns:
            Dict[str, float]: A dictionary with metric names and their evaluated scores.
        """
        pass

    def print_evaluation_results(self, results: Dict[str, float]) -> None:
        """
        Prints the evaluation results in a formatted manner.

        Args:
            results (Dict[str, float]): A dictionary with metric names and their evaluated scores.
        """
        print("Evaluation Results:")
        for metric, value in results.items():
            print(f"{metric}: {value:.2f}")

    def validate_params(self, params: Dict[str, Any], valid_param_sets: List[set]) -> Dict[str, Any]:
        """
        Validates the model parameters against a list of valid parameter sets.

        Args:
            params (Dict[str, Any]): Parameters to validate.
            valid_param_sets (List[set]): A list of sets containing valid parameter names.

        Returns:
            Dict[str, Any]: Validated parameters.

        Raises:
            ValueError: If any invalid parameters are found.
        """
        combined_valid_params = set().union(*valid_param_sets)
        invalid_params = [k for k in params.keys() if k not in combined_valid_params]
        if invalid_params:
            raise ValueError(f"Invalid parameters found: {invalid_params}")
        return {k: v for k, v in params.items() if k in combined_valid_params}


class ClassificationModel(Model):
    """
    Abstract base class for classification models, extending the generic Model class with additional classification-specific methods.
    """

    def balance_train_weights(self, y_train: np.ndarray) -> np.ndarray:
        """
        Balances the training weights based on the class distribution in the training data.

        Args:
            y_train (np.ndarray): Labels for training.

        Returns:
            np.ndarray: Balanced training weights.

        Raises:
            AssertionError: If balancing is attempted on non-binary classification data.
        """
        _, counts = np.unique(y_train, return_counts=True)
        assert len(counts) == 2, 'Only binary classification is supported'
        c_neg, c_pos = counts[0], counts[1]
        pos_weight, neg_weight = 2 * c_neg / (c_neg + c_pos), 2 * c_pos / (c_neg + c_pos)
        train_weights = np.array([pos_weight if label == 1 else neg_weight for label in y_train])
        return train_weights

    @abstractmethod
    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray, y_validation: np.ndarray, training_parameters: Optional[Dict[str, Any]], balance_train_classes: bool) -> None:
        """
        Trains the classification model using provided training and validation data.

        Args:
            x_train (np.ndarray): observations for training.
            y_train (np.ndarray): Labels for training.
            x_validation (np.ndarray): observations for validation.
            y_validation (np.ndarray): Labels for validation.
            training_parameters (Dict[str, Any], optional): Additional training parameters.
            balance_train_classes (bool): Whether to balance the training classes.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, return_proba: bool = False, threshold: float = 0.5) -> np.ndarray:
        """
        Makes predictions for the given input observations.

        Args:
            X (np.ndarray): observations for prediction.
            return_proba (bool, optional): Whether to return probabilities instead of class labels. Defaults to False.
            threshold (float, optional): Threshold for converting probabilities to class labels. Defaults to 0.5.

        Returns:
            np.ndarray: The predicted labels or probabilities.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        pass


class RegressionModel(Model):
    """
    Abstract base class for regression models, providing a framework for training and prediction in regression tasks.
    """

    @abstractmethod
    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray, y_validation: np.ndarray, training_parameters: Optional[Dict[str, Any]]) -> None:
        """
        Trains the regression model using provided training and validation data.

        Args:
            x_train (np.ndarray): observations for training.
            y_train (np.ndarray): Labels for training.
            x_validation (np.ndarray): observations for validation.
            y_validation (np.ndarray): Labels for validation.
            training_parameters (Dict[str, Any], optional): Additional training parameters.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions for the given input observations.

        Args:
            X (np.ndarray): observations for prediction.

        Returns:
            np.ndarray: The predicted values.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        pass

