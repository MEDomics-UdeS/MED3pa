"""
The manager.py module manages the different datasets needed for machine learning workflows, particularly for ``Detectron`` and ``Med3pa`` methods. 
It includes the ``DatasetsManager`` class that contains the training, validation, reference, and testing datasets for a specific ML task.
"""

import numpy as np
import pandas as pd

from .loading_context import DataLoadingContext
from .masked import MaskedDataset


class DatasetsManager:
    """
    Manages various datasets for execution of detectron and med3pa methods.

    This manager is responsible for loading and holding different sets of data, including training, validation,
    reference (or domain dataset), and testing datasets (or new encountered data).
    """
    
    def __init__(self):
        """Initializes the DatasetsManager with empty datasets."""
        self.base_model_training_set = None
        self.base_model_validation_set = None
        self.reference_set = None
        self.testing_set = None
        self.column_labels = None
    
    def set_from_file(self, dataset_type: str, file: str, target_column_name: str) -> None:
        """
        Loads and sets the specified dataset from a file.

        Args:
            dataset_type (str): The type of dataset to set ('training', 'validation', 'reference', 'testing').
            file (str): The file path to the data.
            target_column_name (str): The name of the target column in the dataset.

        Raises:
            ValueError: If an invalid dataset_type is provided or if the shape of observations does not match column labels.
        """
        ctx = DataLoadingContext(file)
        column_labels, obs_np, true_labels_np = ctx.load_as_np(file, target_column_name)
        
        self.set_column_labels(column_labels)

        # Check if the number of columns in observations matches the length of column_labels
        if obs_np.shape[1] != len(self.column_labels):
            raise ValueError(f"The shape of observations {obs_np.shape} does not match the length of column labels {len(column_labels)}")
        
        dataset = MaskedDataset(obs_np, true_labels_np, column_labels=self.column_labels)
        dataset.set_file_path(file=file)
        
        if dataset_type == 'training':
            self.base_model_training_set = dataset
        elif dataset_type == 'validation':
            self.base_model_validation_set = dataset
        elif dataset_type == 'reference':
            self.reference_set = dataset
        elif dataset_type == 'testing':
            self.testing_set = dataset
        else:
            raise ValueError(f"Invalid dataset_type provided: {dataset_type}")

        
    def set_from_data(self, dataset_type: str, observations: np.ndarray, true_labels: np.ndarray, column_labels: list = None) -> None:
        """
        Sets the specified dataset using numpy arrays for observations and true labels.

        Args:
            dataset_type (str): The type of dataset to set ('training', 'validation', 'reference', 'testing').
            observations (np.ndarray): The feature vectors of the dataset.
            true_labels (np.ndarray): The true labels of the dataset.
            column_labels (list, optional): The list of column labels for the dataset. Defaults to None.
        
        Raises:
            ValueError: If an invalid dataset_type is provided or if column labels do not match existing column labels.
            ValueError: If column_labels and target_column_name are not provided when column_labels are not set.
        """
        if column_labels is not None:
            self.set_column_labels(column_labels)
        elif self.column_labels is None:
            raise ValueError("Column labels must be provided when setting a dataset for the first time.")

        dataset = MaskedDataset(observations, true_labels, column_labels=self.column_labels)
        
        if dataset_type == 'training':
            self.base_model_training_set = dataset
        elif dataset_type == 'validation':
            self.base_model_validation_set = dataset
        elif dataset_type == 'reference':
            self.reference_set = dataset
        elif dataset_type == 'testing':
            self.testing_set = dataset
        else:
            raise ValueError(f"Invalid dataset_type provided: {dataset_type}")


    def set_column_labels(self, columns: list) -> None:
        """
        Sets the column labels for the datasets, excluding the target column.

        Args:
            columns (list): The list of columns excluding the target column.

        Raises:
            ValueError: If the target column is not found in the list of columns.
        """

        if self.column_labels is None:
            self.column_labels = columns
        else:
            if self.column_labels != columns:
                raise ValueError("Provided column labels do not match the existing column labels.")
        
        if self.base_model_training_set is not None:
            self.base_model_training_set.column_labels = columns
        if self.base_model_validation_set is not None:
            self.base_model_validation_set.column_labels = columns
        if self.reference_set is not None:
            self.reference_set.column_labels = columns
        if self.testing_set is not None:
            self.testing_set.column_labels = columns

    def get_column_labels(self):
        """
        Retrieves the column labels of the manager
        
        Returns:
            List[str]: A list of the column labels extracted from the files.

        """
        return self.column_labels
    
    def get_info(self, show_details: bool = True) -> dict:
        """
        Returns information about all the datasets managed by the DatasetsManager.

        Args:
            detailed (bool): If True, includes detailed information about each dataset. If False, only indicates whether each dataset is set.

        Returns:
            dict: A dictionary containing information about each dataset.
        """
        if show_details:
            datasets_info = {
                'training_set': self.base_model_training_set.get_info() if self.base_model_training_set else 'Not set',
                'validation_set': self.base_model_validation_set.get_info() if self.base_model_validation_set else 'Not set',
                'reference_set': self.reference_set.get_info() if self.reference_set else 'Not set',
                'testing_set': self.testing_set.get_info() if self.testing_set else 'Not set',
                'column_labels': self.column_labels if self.column_labels else 'Not set'
            }
        else:
            datasets_info = {
                'training_set': 'Set' if self.base_model_training_set else 'Not set',
                'validation_set': 'Set' if self.base_model_validation_set else 'Not set',
                'reference_set': 'Set' if self.reference_set else 'Not set',
                'testing_set': 'Set' if self.testing_set else 'Not set',
                'column_labels': 'Set' if self.column_labels else 'Not set'
            }
        return datasets_info
    
    def summarize(self) -> None:
        """
        Prints a summary of the manager.
        """
        info = self.get_info()
        print(f"training_set: {info['training_set']}")
        print(f"validation_set: {info['validation_set']}")
        print(f"reference_set: {info['reference_set']}")
        print(f"testing_set: {info['testing_set']}")
        print(f"column_labels: {info['column_labels']}")

    def reset_datasets(self) -> None:
        """
        Resets all datasets in the manager.
        """
        self.base_model_training_set = None
        self.base_model_validation_set = None
        self.reference_set = None
        self.testing_set = None
        self.column_labels = None

    def get_dataset_by_type(self, dataset_type: str, return_instance: bool = False) -> MaskedDataset:
        """
        Helper method to get a dataset by type.

        Args:
            dataset_type (str): The type of dataset to retrieve ('training', 'validation', 'reference', 'testing').

        Returns:
            MaskedDataset: The corresponding MaskedDataset instance.
        
        Raises:
            ValueError: If an invalid dataset_type is provided.
        """
        if dataset_type == 'training':
            return self.__get_base_model_training_data(return_instance=return_instance)
        elif dataset_type == 'validation':
            return self.__get_base_model_validation_data(return_instance=return_instance)
        elif dataset_type == 'reference':
            return self.__get_reference_data(return_instance=return_instance)
        elif dataset_type == 'testing':
            return self.__get_testing_data(return_instance=return_instance)
        else:
            raise ValueError(f"Invalid dataset_type provided: {dataset_type}")

    def save_dataset_to_csv(self, dataset_type: str, file_path: str) -> None:
        """
        Saves the specified dataset to a CSV file.

        Args:
            dataset_type (str): The type of dataset to save ('training', 'validation', 'reference', 'testing').
            file_path (str): The file path to save the dataset to.
        
        Raises:
            ValueError: If an invalid dataset_type is provided.
        """
        dataset = self.get_dataset_by_type(dataset_type, True)
        if dataset is None:
            raise ValueError(f"Dataset '{dataset_type}' is not set.")
        
        dataset.save_to_csv(file_path)

    def __get_base_model_training_data(self, return_instance: bool = False):
        """
        Retrieves the training dataset.

        Args:
            return_instance (bool, optional): If True, returns the MaskedDataset instance; otherwise, returns the observations and true labels. Defaults to False.

        Returns:
            Union[tuple, MaskedDataset]: The observations and true labels if return_instance is False, otherwise the MaskedDataset instance.

        Raises:
            ValueError: If the base model training set is not initialized.
        """
        if self.base_model_training_set is not None:
            if return_instance:
                return self.base_model_training_set
            return self.base_model_training_set.get_observations(), self.base_model_training_set.get_true_labels()
        else:
            raise ValueError("Base model training set not initialized.")

    def __get_base_model_validation_data(self, return_instance: bool = False):
        """
        Retrieves the validation dataset.

        Args:
            return_instance (bool, optional): If True, returns the MaskedDataset instance; otherwise, returns the observations and true labels. Defaults to False.

        Returns:
            Union[tuple, MaskedDataset]: The observations and true labels if return_instance is False, otherwise the MaskedDataset instance.

        Raises:
            ValueError: If the base model validation set is not initialized.
        """
        if self.base_model_validation_set is not None:
            if return_instance:
                return self.base_model_validation_set
            return self.base_model_validation_set.get_observations(), self.base_model_validation_set.get_true_labels()
        else:
            raise ValueError("Base model validation set not initialized.")

    def __get_reference_data(self, return_instance: bool = False):
        """
        Retrieves the reference dataset.

        Args:
            return_instance (bool, optional): If True, returns the MaskedDataset instance; otherwise, returns the observations and true labels. Defaults to False.

        Returns:
            Union[tuple, MaskedDataset]: The observations and true labels if return_instance is False, otherwise the MaskedDataset instance.

        Raises:
            ValueError: If the reference set is not initialized.
        """
        if self.reference_set is not None:
            if return_instance:
                return self.reference_set
            return self.reference_set.get_observations(), self.reference_set.get_true_labels()
        else:
            raise ValueError("Reference set not initialized.")

    def __get_testing_data(self, return_instance: bool = False):
        """
        Retrieves the testing dataset.

        Args:
            return_instance (bool, optional): If True, returns the MaskedDataset instance; otherwise, returns the observations and true labels. Defaults to False.

        Returns:
            Union[tuple, MaskedDataset]: The observations and true labels if return_instance is False, otherwise the MaskedDataset instance.

        Raises:
            ValueError: If the testing set is not initialized.
        """
        if self.testing_set is not None:
            if return_instance:
                return self.testing_set
            return self.testing_set.get_observations(), self.testing_set.get_true_labels()
        else:
            raise ValueError("Testing set not initialized.")

