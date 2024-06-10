"""
This module provides strategies for preparing data in different formats for model training and prediction.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb


class DataPreparingStrategy:
    """
    Abstract base class for data preparation strategies.
    """

    @staticmethod
    def execute(features, labels=None, weights=None):
        """
        Prepares data for model training or prediction.

        Args:
            features (array-like): Features array.
            labels (array-like, optional): Labels array.
            weights (array-like, optional): Weights array.

        Returns:
            object: Prepared data in the required format for the model.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class ToDmatrixStrategy(DataPreparingStrategy):
    """
    Concrete implementation for converting data into DMatrix format suitable for XGBoost models.
    """

    @staticmethod
    def is_supported_data(features, labels=None, weights=None) -> bool:
        """
        Checks if the data types of features, labels, and weights are supported for conversion to DMatrix.

        Args:
            features (array-like): Features data.
            labels (array-like, optional): Labels data.
            weights (array-like, optional): Weights data.

        Returns:
            bool: True if all data types are supported, False otherwise.
        """
        supported_types = [np.ndarray, pd.DataFrame, sp.spmatrix, list]
        is_supported = lambda data: any(isinstance(data, t) for t in supported_types)

        return all(is_supported(data) for data in [features, labels, weights] if data is not None)

    @staticmethod
    def execute(features, labels=None, weights=None) -> xgb.DMatrix:
        """
        Converts features, labels, and weights into an XGBoost DMatrix.

        Args:
            features (array-like): Features data.
            labels (array-like, optional): Labels data.
            weights (array-like, optional): Weights data.

        Returns:
            xgb.DMatrix: A DMatrix object ready for use with XGBoost.

        Raises:
            ValueError: If any input data types are not supported.
        """
        if not ToDmatrixStrategy.is_supported_data(features, labels, weights):
            raise ValueError("Unsupported data type provided for creating DMatrix.")
        return xgb.DMatrix(data=features, label=labels, weight=weights)


class ToNumpyStrategy(DataPreparingStrategy):
    """
    Converts input data to NumPy arrays, ensuring compatibility with models expecting NumPy inputs.
    """

    @staticmethod
    def execute(features, labels=None, weights=None) -> tuple:
        """
        Converts features, labels, and weights into NumPy arrays.

        Args:
            features (array-like): Features data.
            labels (array-like, optional): Labels data.
            weights (array-like, optional): Weights data.

        Returns:
            tuple: A tuple of NumPy arrays for features, labels, and weights. Returns None for labels and weights if they are not provided.

        Raises:
            ValueError: If the features or labels are empty arrays.
        """
        features_np = np.asarray(features)
        labels_np = np.asarray(labels) if labels is not None else None
        weights_np = np.asarray(weights) if weights is not None else None

        if features_np.size == 0:
            raise ValueError("Features array cannot be empty.")
        if labels is not None and labels_np.size == 0:
            raise ValueError("Labels array cannot be empty.")

        return features_np, labels_np, weights_np


class ToDataframesStrategy(DataPreparingStrategy):
    """
    Converts input data to pandas DataFrames, suitable for models requiring DataFrame inputs.
    """

    @staticmethod
    def execute(column_labels: list, features: np.ndarray, labels: np.ndarray = None, weights: np.ndarray = None) -> tuple:
        """
        Converts features, labels, and weights into pandas DataFrames with specified column labels.

        Args:
            column_labels (list): Column labels for the features DataFrame.
            features (np.ndarray): Features array.
            labels (np.ndarray, optional): Labels array.
            weights (np.ndarray, optional): Weights array.

        Returns:
            tuple: DataFrames for features, labels, and weights. Returns None for labels and weights DataFrames if not provided.

        Raises:
            ValueError: If the features array is empty.
        """
        if features.size == 0:
            raise ValueError("Features array cannot be empty.")

        X_df = pd.DataFrame(features, columns=column_labels)
        Y_df = pd.DataFrame(labels, columns=['Label']) if labels is not None else None
        W_df = pd.DataFrame(weights, columns=['Weight']) if weights is not None else None

        return X_df, Y_df, W_df
