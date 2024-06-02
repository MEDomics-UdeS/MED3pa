"""
Module: data_preparation_strategies
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

        Parameters
        ----------
        features : array-like
            Features array.
        labels : array-like, optional
            Labels array.
        weights : array-like, optional
            Weights array.

        Returns
        -------
        object
            Prepared data in the required format for the model.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class ToDmatrixStrategy(DataPreparingStrategy):
    """
    Concrete implementation for converting data into DMatrix format.
    """
    @staticmethod
    def is_supported_data(features, labels=None, weights=None) -> bool:
        """
        Check if the features, labels, and weights are of supported types for creating a DMatrix.

        Parameters
        ----------
        features : array-like
            Features array or supported format.
        labels : array-like, optional
            Labels array or supported format.
        weights : array-like, optional
            Weights array or supported format.

        Returns
        -------
        bool
            True if supported, False otherwise.
        """
        supported_types = [np.ndarray, pd.DataFrame, sp.spmatrix, list]

        if not any(isinstance(features, t) for t in supported_types):
            return False

        if weights is not None and not any(isinstance(weights, t) for t in supported_types):
            return False

        if labels is not None and not any(isinstance(labels, t) for t in supported_types):
            return False

        return True

    @staticmethod
    def execute(features, labels=None, weights=None) -> xgb.DMatrix:
        """
        Converts features, labels, and weights into a DMatrix object, handling various input types.

        Parameters
        ----------
        features : array-like
            Features array or supported format.
        labels : array-like, optional
            Labels array or supported format.
        weights : array-like, optional
            Weights array or supported format.

        Returns
        -------
        xgb.DMatrix
            A DMatrix object containing the features, labels, and weights.

        Raises
        ------
        ValueError
            If unsupported data type is provided for creating DMatrix.
        """
        if not ToDmatrixStrategy.is_supported_data(features, labels, weights):
            raise ValueError("Unsupported data type provided for creating DMatrix.")

        return xgb.DMatrix(data=features, label=labels, weight=weights)


class ToNumpyStrategy(DataPreparingStrategy):
    """
    Concrete implementation of DataPreparingStrategy for converting data into NumPy array format.
    """
    @staticmethod
    def execute(features, labels=None, weights=None) -> tuple:
        """
        Converts features, labels, and weights into NumPy arrays.

        Parameters
        ----------
        features : array-like
            Features array or any format that can be converted to a NumPy array.
        labels : array-like, optional
            Labels array or any format that can be converted to a NumPy array.
        weights : array-like, optional
            Weights array or any format that can be converted to a NumPy array.

        Returns
        -------
        tuple
            A tuple containing the features, labels, and weights as NumPy arrays. If labels are not provided, None is returned for labels.

        Raises
        ------
        ValueError
            If the features or labels array is empty.
        """
        features_np = np.asarray(features)
        if labels is None:
            return features_np, None, None
        labels_np = np.asarray(labels)
        weights_np = np.asarray(weights) if weights is not None else None

        if features_np.size == 0:
            raise ValueError("Cannot build a NumPy array from an empty features array.")
        if labels is not None and labels_np.size == 0:
            raise ValueError("Cannot build a NumPy array from an empty labels array.")

        return features_np, labels_np, weights_np


class ToDataframesStrategy(DataPreparingStrategy):
    """
    Concrete implementation of DataPreparingStrategy for converting data into pandas DataFrame format.
    """
    @staticmethod
    def execute(column_labels: list, features: np.ndarray, labels: np.ndarray = None, weights: np.ndarray = None) -> tuple:
        """
        Converts features, labels, and weights into pandas DataFrames with specified column labels for features.

        Parameters
        ----------
        column_labels : list
            List containing column labels for features.
        features : np.ndarray
            NumPy array representing the features.
        labels : np.ndarray, optional
            NumPy array representing the target values.
        weights : np.ndarray, optional
            NumPy array representing the weights.

        Returns
        -------
        tuple
            A tuple containing the features DataFrame, labels DataFrame, and weights DataFrame.
        """
        X_df = pd.DataFrame(features, columns=column_labels)

        if labels is not None:
            Y_df = pd.DataFrame(labels)
        else:
            Y_df = None

        W_df = pd.DataFrame(weights) if weights is not None else None

        return X_df, Y_df, W_df
