"""
This module provides a flexible framework for loading datasets from various file formats by utilizing the **strategy design pattern**.
It supports dynamic selection of data loading strategies based on the file extension, enabling easy extension and maintenance.
It includes the ``DataLoadingContext`` class, responsible for selecting and setting the right **loading strategy** based on the loaded file extension.
"""
import numpy as np
from typing import Tuple, List

from .loading_strategies import DataLoadingStrategy, CSVDataLoadingStrategy


class DataLoadingContext:
    """
    A context class for managing data loading strategies. It supports setting and getting the current
    data loading strategy, as well as loading data as a NumPy array from a specified file.
    """

    strategies = {
        'csv': CSVDataLoadingStrategy,
    }

    def __init__(self, file_path: str):
        """
        Initializes the data loading context with a strategy based on the file extension.

        Args:
            file_path (str): The path to the dataset file.

        Raises:
            ValueError: If the file extension is not supported.
        """
        file_extension = file_path.split('.')[-1]
        strategy_class = self.strategies.get(file_extension, None)
        if strategy_class is None:
            raise ValueError(f"This file extension is not supported yet: '{file_extension}'")
        self.selected_strategy = strategy_class()

    def set_strategy(self, strategy: DataLoadingStrategy) -> None:
        """
        Sets a new data loading strategy.

        Args:
            strategy (DataLoadingStrategy): The new data loading strategy to be used.
        """
        self.selected_strategy = strategy

    def get_strategy(self) -> DataLoadingStrategy:
        """
        Returns the currently selected data loading strategy.

        Returns:
            DataLoadingStrategy: The currently selected data loading strategy.
        """
        return self.selected_strategy

    def load_as_np(self, file_path: str, target_column_name: str) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Loads data from the given file path and returns it as a NumPy array, along with column labels and the target data.

        Args:
            file_path (str): The path to the dataset file.
            target_column_name (str): The name of the target column, such as true labels or values in case of regression.

        Returns:
            Tuple[List[str], np.ndarray, np.ndarray]: A tuple containing the column labels, observations as a NumPy array, 
            and the target as a NumPy array.
        """
        return self.selected_strategy.execute(file_path, target_column_name)


def supported_file_formats() -> List[str]:
    """
    Returns a list of supported file formats.

    Returns:
        List[str]: A list of supported file formats.
    """
    return list(DataLoadingContext.strategies.keys())
