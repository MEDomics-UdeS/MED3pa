"""
Module: data_loading_strategies
This module provides different strategies for loading datasets.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple

class DataLoadingStrategy:
    """
    Abstract base class for data loading strategies.

    Methods
    -------
    execute(path_to_file: str, target_column_name: str) -> Tuple[List[str], np.ndarray, np.ndarray]
        Abstract method to execute the data loading strategy.
    """

    @staticmethod
    def execute(path_to_file: str, target_column_name: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Abstract method to execute the data loading strategy.

        Parameters
        ----------
        path_to_file : str
            The path to the file to be loaded.
        target_column_name : str
            The name of the target column in the dataset.

        Returns
        -------
        Tuple[List[str], np.ndarray, np.ndarray]
            Column labels, features as NumPy array, and target as NumPy array.
        """
        raise NotImplementedError("This method should be overridden by subclasses")

class CSVDataLoadingStrategy(DataLoadingStrategy):
    """
    Strategy for loading CSV data.

    Methods
    -------
    execute(path_to_file: str, target_column_name: str) -> Tuple[List[str], np.ndarray, np.ndarray]
        Loads CSV data from the given path, separates features and target, and converts them to NumPy arrays.
    """

    @staticmethod
    def execute(path_to_file: str, target_column_name: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Loads CSV data from the given path, separates features and target, and converts them to NumPy arrays.

        Parameters
        ----------
        path_to_file : str
            The path to the CSV file to be loaded.
        target_column_name : str
            The name of the target column in the dataset.

        Returns
        -------
        Tuple[List[str], np.ndarray, np.ndarray]
            Column labels, features as NumPy array, and target as NumPy array.
        """
        # Read the CSV file
        df = pd.read_csv(path_to_file)
        
        # Separate features and target
        features = df.drop(columns=[target_column_name])  
        target = df[target_column_name]  
        column_labels = features.columns.tolist()

        # Convert to NumPy arrays
        features_np = features.to_numpy()
        target_np = target.to_numpy()
        
        return column_labels, features_np, target_np
