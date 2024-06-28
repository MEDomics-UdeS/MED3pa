"""
This module is crucial for data handling, utilizing the **Strategy design pattern** and therefor offering multiple strategies to transform raw data into formats that enhance model training and evaluation.
According to the model type.
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
    def execute(observations, labels=None, weights=None):
        """
        Prepares data for model training or prediction.

        Args:
            observations (array-like): observations array.
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
    def is_supported_data(observations, labels=None, weights=None) -> bool:
        """
        Checks if the data types of observations, labels, and weights are supported for conversion to DMatrix.

        Args:
            observations (array-like): observations data.
            labels (array-like, optional): Labels data.
            weights (array-like, optional): Weights data.

        Returns:
            bool: True if all data types are supported, False otherwise.
        """
        supported_types = [np.ndarray, pd.DataFrame, sp.spmatrix, list]
        is_supported = lambda data: any(isinstance(data, t) for t in supported_types)

        return all(is_supported(data) for data in [observations, labels, weights] if data is not None)

    @staticmethod
    def execute(observations, labels=None, weights=None) -> xgb.DMatrix:
        """
        Converts observations, labels, and weights into an XGBoost DMatrix.

        Args:
            observations (array-like): observations data.
            labels (array-like, optional): Labels data.
            weights (array-like, optional): Weights data.

        Returns:
            xgb.DMatrix: A DMatrix object ready for use with XGBoost.

        Raises:
            ValueError: If any input data types are not supported.
        """
        if not ToDmatrixStrategy.is_supported_data(observations, labels, weights):
            raise ValueError("Unsupported data type provided for creating DMatrix.")
        return xgb.DMatrix(data=observations, label=labels, weight=weights)


class ToNumpyStrategy(DataPreparingStrategy):
    """
    Converts input data to NumPy arrays, ensuring compatibility with models expecting NumPy inputs.
    """

    @staticmethod
    def execute(observations, labels=None, weights=None) -> tuple:
        """
        Converts observations, labels, and weights into NumPy arrays.

        Args:
            observations (array-like): observations data.
            labels (array-like, optional): Labels data.
            weights (array-like, optional): Weights data.

        Returns:
            tuple: A tuple of NumPy arrays for observations, labels, and weights. Returns None for labels and weights if they are not provided.

        Raises:
            ValueError: If the observations or labels are empty arrays.
        """
        obs_np = np.asarray(observations)
        labels_np = np.asarray(labels) if labels is not None else None
        weights_np = np.asarray(weights) if weights is not None else None

        if obs_np.size == 0:
            raise ValueError("Observations array cannot be empty.")
        if labels is not None and labels_np.size == 0:
            raise ValueError("Labels array cannot be empty.")

        return obs_np, labels_np, weights_np


class ToDataframesStrategy(DataPreparingStrategy):
    """
    Converts input data to pandas DataFrames, suitable for models requiring DataFrame inputs.
    """

    @staticmethod
    def execute(column_labels: list, observations: np.ndarray, labels: np.ndarray = None, weights: np.ndarray = None) -> tuple:
        """
        Converts observations, labels, and weights into pandas DataFrames with specified column labels.

        Args:
            column_labels (list): Column labels for the observations DataFrame.
            observations (np.ndarray): observations array.
            labels (np.ndarray, optional): Labels array.
            weights (np.ndarray, optional): Weights array.

        Returns:
            tuple: DataFrames for observations, labels, and weights. Returns None for labels and weights DataFrames if not provided.

        Raises:
            ValueError: If the observations array is empty.
        """
        if observations.size == 0:
            raise ValueError("observations array cannot be empty.")

        X_df = pd.DataFrame(observations, columns=column_labels)
        Y_df = pd.DataFrame(labels, columns=['Label']) if labels is not None else None
        W_df = pd.DataFrame(weights, columns=['Weight']) if weights is not None else None

        return X_df, Y_df, W_df
