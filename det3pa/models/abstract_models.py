import numpy as np
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod
from .data_strategies import DataPreparingStrategy

class Model(ABC):
    """
    Abstract base class for models. Defines the structure that all models should follow.
    """

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray, eval_metrics: List[str], print_results: bool = False) -> Dict[str, float]:
        """
        Evaluates the model using specified metrics.

        Parameters
        ----------
        X : np.ndarray
            Features for evaluation.
        y : np.ndarray
            True labels for evaluation.
        eval_metrics : list of str
            Metrics to use for evaluation.
        print_results : bool, optional
            Whether to print the evaluation results. Defaults to False.

        Returns
        -------
        dict
            A dictionary with metric names and their evaluated scores.
        """
        pass

    def print_evaluation_results(self, results: Dict[str, float]) -> None:
        """
        Prints the evaluation results.

        Parameters
        ----------
        results : dict
            A dictionary with metric names and their evaluated scores.
        """
        print("Evaluation Results:")
        for metric, value in results.items():
            print(f"{metric}: {value:.2f}")

    def validate_params(self, params: Dict[str, Any], valid_param_sets: List[set]) -> Dict[str, Any]:
        """
        Validates and returns the parameters.

        Parameters
        ----------
        params : dict
            Dictionary of parameters to validate.
        valid_param_sets : list of sets
            List of sets containing valid parameter names.

        Returns
        -------
        dict
            Validated parameters.

        Raises
        ------
        ValueError
            If any invalid parameters are found.
        """
        combined_valid_params = set().union(*valid_param_sets)
        invalid_params = [k for k in params.keys() if k not in combined_valid_params]
        if invalid_params:
            raise ValueError(f"Invalid parameters found: {invalid_params}")
        return {k: v for k, v in params.items() if k in combined_valid_params}


    def get_data_strategy(self) -> DataPreparingStrategy:
        """
        Returns the data preparation strategy.
        """
        return self.data_preparation_strategy

    def set_model(self, model) -> None:
        """
        Sets the model instance.

        Parameters
        ----------
        model : Any
            The model to set.
        """
        self.model = model
        self.model_class = type(model)

class ClassificationModel(Model):
    """
    Abstract base class for classification models.
    """
    def balance_train_weights(self, y_train: np.ndarray) -> np.ndarray:
        """
        Balances the training weights based on class distribution.

        Parameters
        ----------
        y_train : np.ndarray
            Labels for training.

        Returns
        -------
        np.ndarray
            Balanced training weights.
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
        Trains the model on the given dataset.

        Parameters
        ----------
        x_train : np.ndarray
            Features for training.
        y_train : np.ndarray
            Labels for training.
        x_validation : np.ndarray
            Features for validation.
        y_validation : np.ndarray
            Labels for validation.
        training_parameters : dict, optional
            Additional training parameters.
        balance_train_classes : bool
            Whether to balance the training classes.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, return_proba: bool = False, threshold: float = 0.5) -> np.ndarray:
        """
        Makes predictions using the model for the given input.

        Parameters
        ----------
        X : np.ndarray
            Features for prediction.
        return_proba : bool, optional
            Whether to return probabilities. Defaults to False.
        threshold : float, optional
            Threshold for converting probabilities to class labels. Defaults to 0.5.

        Returns
        -------
        np.ndarray
            Predictions made by the model.
        """
        pass

    @abstractmethod
    def train_to_disagree(self, x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray, y_validation: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, training_parameters: Optional[Dict[str, Any]], balance_train_classes: bool, N: int) -> None:
        """
        Trains the model to disagree with another model.

        Parameters
        ----------
        x_train : np.ndarray
            Features for training.
        y_train : np.ndarray
            Labels for training.
        x_validation : np.ndarray
            Features for validation.
        y_validation : np.ndarray
            Labels for validation.
        x_test : np.ndarray
            Features to disagree on.
        y_test : np.ndarray
            Labels to disagree on.
        training_parameters : dict, optional
            Additional training parameters.
        balance_train_classes : bool
            Whether to balance the training classes.
        N : int
            Number of examples to use for disagreement training.
        """
        pass

class RegressionModel(Model):
    """
    Abstract base class for regression models.
    """

    @abstractmethod
    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray, y_validation: np.ndarray, training_parameters: Optional[Dict[str, Any]]) -> None:
        """
        Trains the model on the given dataset.

        Parameters
        ----------
        x_train : np.ndarray
            Features for training.
        y_train : np.ndarray
            Labels for training.
        x_validation : np.ndarray
            Features for validation.
        y_validation : np.ndarray
            Labels for validation.
        training_parameters : dict, optional
            Additional training parameters.
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions using the model for the given input.

        Parameters
        ----------
        X : np.ndarray
            Features for prediction.

        Returns
        -------
        np.ndarray
            Predictions made by the model.
        """
        pass
