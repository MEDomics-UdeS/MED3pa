"""
This module provides strategies for loading data from files into usable Python formats, focusing on converting data into **NumPy** arrays. 
It includes an abstract base class ``DataLoadingStrategy`` for defining common interfaces and concrete implementations of this class, such as ``CSVDataLoadingStrategy`` for handling CSV files.
This setup allows easy extension to support additional file types as needed.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from abc import ABC, abstractmethod


class DataLoadingStrategy(ABC):
    """
    Abstract base class for data loading strategies. Defines a common interface for all data loading strategies.
    """
    @abstractmethod
    def execute(path_to_file: str, target_column_name: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Abstract method to execute the data loading strategy.

        Args:
            path_to_file (str): The path to the file to be loaded.
            target_column_name (str): The name of the target column in the dataset.

        Returns:
            Tuple[List[str], np.ndarray, np.ndarray]: A tuple containing the column labels, observations as a NumPy array, 
            and the target as a NumPy array.

        """
        pass


class CSVDataLoadingStrategy(DataLoadingStrategy):
    """
    Strategy class for loading CSV data. Implements the abstract execute method to handle CSV files.

    Methods:
        execute(path_to_file: str, target_column_name: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
            Loads CSV data from the given path, separates observations and target, and converts them to NumPy arrays.
    """

    @staticmethod
    def execute(path_to_file: str, target_column_name: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Loads CSV data from the given path, separates observations and target, and converts them to NumPy arrays.

        Args:
            path_to_file (str): The path to the CSV file to be loaded.
            target_column_name (str): The name of the target column in the dataset.

        Returns:
            Tuple[List[str], np.ndarray, np.ndarray]: Column labels, observations as a NumPy array, and target as a NumPy array.
        """
        # Read the CSV file
        df = pd.read_csv(path_to_file)
        
        # Separate observations and target
        observations = df.drop(columns=[target_column_name])  
        target = df[target_column_name]  
        column_labels = observations.columns.tolist()

        # Convert to NumPy arrays
        obs_np = observations.to_numpy()
        target_np = target.to_numpy()
        
        return column_labels, obs_np, target_np


