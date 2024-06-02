"""
Module: models
This module provides implementations of different machine learning models, including XGBoost, RandomForestRegressor, and DecisionTreeRegressor.
"""

import xgboost as xgb
import numpy as np
from .data_strategies import ToDmatrixStrategy
from .xgboost_params import valid_xgboost_params, xgboost_metrics, valid_xgboost_custom_params
from .classification_metrics import *
from .abstract_models import ClassificationModel
from typing import Union, Optional, List, Dict, Any


class XGBoostModel(ClassificationModel):
    """
    A concrete implementation of the Model class for XGBoost models.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None, model: Optional[Union[xgb.Booster, xgb.XGBClassifier]] = None, model_class: Optional[Any] = None) -> None:
        """
        Initializes the XGBoostModel either with parameters for a new model or a loaded pickled model.

        Parameters
        ----------
        params : dict, optional
            A dictionary of parameters for the booster model.
        model : xgb.Booster or xgb.XGBClassifier, optional
            A loaded pickled model.
        model_class : Any, optional
            Specifies the class of the model, xgb.XGBClassifier or xgb.Booster.
        """
        self.params = params
        self.model = model
        self.model_class = model_class if model_class else xgb.Booster
        self.pickled_model = model is not None
        self.data_preparation_strategy = ToDmatrixStrategy()

    def _ensure_dmatrix(self, features: Any, labels: Optional[np.ndarray] = None, weights: Optional[np.ndarray] = None) -> xgb.DMatrix:
        """
        Ensures that the input data is converted to a DMatrix format, using the defined data preparation strategy.

        Parameters
        ----------
        features : Any
            Features array.
        labels : np.ndarray, optional
            Labels array.
        weights : np.ndarray, optional
            Weights array.

        Returns
        -------
        xgb.DMatrix
            A DMatrix object.
        """
        if not isinstance(features, xgb.DMatrix):
            return self.data_preparation_strategy.execute(features, labels, weights)
        else:
            return features

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray, y_validation: np.ndarray, training_parameters: Optional[Dict[str, Any]], balance_train_classes: bool) -> None:
        """
        Trains the model on the provided dataset.

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

        Raises
        ------
        ValueError
            If parameters for xgb.Booster are not initialized before training.
        NotImplementedError
            If the model_class is not supported for training.
        """
        if training_parameters:
            valid_param_sets = [valid_xgboost_params, valid_xgboost_custom_params]
            valid_training_params = self.validate_params(training_parameters, valid_param_sets)
            self.params.update(valid_training_params)
            weights = self.balance_train_weights(y_train) if balance_train_classes else training_parameters.get('training_weights', np.ones_like(y_train))
            evaluation_metrics = training_parameters.get('custom_eval_metrics', self.params.get('eval_metric', ["Accuracy"]) if self.params else ["Accuracy"])
            num_boost_rounds = training_parameters.get('num_boost_rounds', self.params.get('num_boost_rounds', 10) if self.params else 10)

        else:
            weights = np.ones_like(y_train)
            evaluation_metrics = self.params.get('eval_metric', ["Accuracy"]) if self.params else ["Accuracy"]
            num_boost_rounds = self.params.get('num_boost_rounds', 10) if self.params else 10

        if not self.params:
            raise ValueError("Parameters must be initialized before training.")
        
        filtered_params = {k: v for k, v in self.params.items() if k not in valid_xgboost_custom_params}

        if self.model_class is xgb.Booster:
            if not self.params:
                raise ValueError("Parameters for xgb.Booster must be initialized before training.")
            dtrain = self._ensure_dmatrix(x_train, y_train, weights)
            dval = self._ensure_dmatrix(x_validation, y_validation)
            self.model = xgb.train(filtered_params, dtrain, num_boost_round=num_boost_rounds, evals=[(dval, 'eval')], verbose_eval=False)
            self.evaluate(x_validation, y_validation, eval_metrics=evaluation_metrics, print_results=True)
        elif self.model_class is xgb.XGBClassifier:
            self.model = self.model_class(**filtered_params)
            self.model.fit(x_train, y_train, sample_weight=weights, eval_set=[(x_validation, y_validation)])
            self.evaluate(x_validation, y_validation, eval_metrics=evaluation_metrics, print_results=True)
        else:
            raise NotImplementedError(f"Training not implemented for model class {self.model_class}")

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

        Raises
        ------
        ValueError
            If the model has not been initialized.
        NotImplementedError
            If prediction is not implemented for the model class.
        """
        if self.model is None:
            raise ValueError(f"The {self.model_class.__name__} model has not been initialized.")

        if self.model_class is xgb.Booster:
            dtest = self._ensure_dmatrix(X)
            preds = self.model.predict(dtest)
        elif self.model_class is xgb.XGBClassifier:
            preds = self.model.predict_proba(X) if return_proba else self.model.predict(X)
        else:
            raise NotImplementedError(f"Prediction not implemented for model class {self.model_class}")

        return preds if return_proba else (preds > threshold).astype(int)

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
            Features for testing.
        y_test : np.ndarray
            Labels for testing.
        training_parameters : dict, optional
            Additional training parameters.
        balance_train_classes : bool
            Whether to balance the training classes.
        N : int
            Number of examples to use for disagreement training.

        Raises
        ------
 ValueError
            If parameters for xgb.Booster are not initialized before training.
        NotImplementedError
            If the model_class is not supported for training.
        """
        if training_parameters:
            valid_param_sets = [valid_xgboost_params, valid_xgboost_custom_params]
            valid_training_params = self.validate_params(training_parameters, valid_param_sets)
            self.params.update(valid_training_params)
            training_weights = self.balance_train_weights(y_train) if balance_train_classes else training_parameters.get('training_weights', np.ones_like(y_train))
            evaluation_metrics = training_parameters.get('custom_eval_metrics', self.params.get('eval_metric', ["Accuracy"]) if self.params else ["Accuracy"])
            num_boost_rounds = training_parameters.get('num_boost_rounds', self.params.get('num_boost_rounds', 10) if self.params else 10)
        else:
            training_weights = np.ones_like(y_train)
            evaluation_metrics = self.params.get('eval_metric', ["Accuracy"]) if self.params else ["Accuracy"]
            num_boost_rounds = self.params.get('num_boost_rounds', 10) if self.params else 10
            
        if not self.params:
            raise ValueError("Parameters must be initialized before training.")

        filtered_params = {k: v for k, v in self.params.items() if k not in valid_xgboost_custom_params}

        data = np.concatenate([x_train, x_test])
        label = np.concatenate([y_train, 1 - y_test])
        weight = np.concatenate([training_weights, 1 / (N + 1) * np.ones(N)])

        if self.model_class is xgb.Booster:
            if not self.params:
                raise ValueError("Parameters for xgb.Booster must be initialized before training.")
            dtrain = self._ensure_dmatrix(data, label, weight)
            dval = self._ensure_dmatrix(x_validation, y_validation)
            self.model = xgb.train(filtered_params, dtrain, num_boost_round=num_boost_rounds, evals=[(dval, 'eval')], verbose_eval=False)
            self.evaluate(x_validation, y_validation, eval_metrics=evaluation_metrics)
        elif self.model_class is xgb.XGBClassifier:
            self.model = self.model_class(**filtered_params)
            self.model.fit(data, label, sample_weight=weight, eval_set=[(x_validation, y_validation)])
            self.evaluate(x_validation, y_validation, eval_metrics=evaluation_metrics)
        else:
            raise NotImplementedError(f"Training not implemented for model class {self.model_class}")

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

        Raises
        ------
        ValueError
            If the model has not been trained before evaluation.
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation.")

        probs = self.predict(X, return_proba=True)
        if probs.ndim == 1 :
            preds = (probs > 0.5).astype(int)
        else:
            preds = None

        if preds is None:
            raise ValueError("Only binary classification is supported for this version.")

        evaluation_results = {}
        for metric_name in eval_metrics:
            translated_metric_name = xgboost_metrics.get(metric_name, metric_name)
            metric = metrics_mappings.get(translated_metric_name, None)
            if metric:
                evaluation_results[metric_name] = metric.calculate(y, probs) if metric in {RocAuc, AveragePrecision} else metric.calculate(y, preds)
            else:
                print(f"Error: The metric '{metric_name}' is not supported.")

        if print_results:
            self.print_evaluation_results(results=evaluation_results)
        return evaluation_results