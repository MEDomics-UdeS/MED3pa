"""
The manager.py module manages datasets for machine learning workflows, particularly for ``Detectron`` and ``Med3pa`` methods. 
It includes the ``DatasetsManager`` class for handling training, validation, reference, and testing datasets, and the ``MaskedDataset`` class for different data operations.
"""

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from .loading_context import DataLoadingContext

class MaskedDataset(Dataset):
    """
    A dataset wrapper for PyTorch that supports masking and sampling of data points.
    """
    
    def __init__(self, features: np.ndarray, true_labels: np.ndarray, column_labels: list = None):
        """
        Initializes the MaskedDataset.

        Args:
            features (np.ndarray): The feature vectors of the dataset.
            true_labels (np.ndarray): The true labels of the dataset.
            column_labels (list, optional): The column labels for the feature vectors. Defaults to None.
        """
        self.features = features
        self.true_labels = true_labels
        self.indices = np.arange(len(self.features))
        self.original_indices = self.indices.copy()
        self.sample_counts = np.zeros(len(features), dtype=int)
        self.pseudo_probabilities = None
        self.pseudo_labels = None 
        self.confidence_scores = None
        self.column_labels = column_labels if column_labels is not None else [f'feature_{i}' for i in range(features.shape[1])]

    def __getitem__(self, index: int) -> tuple:
        """
        Retrieves the data point and its label(s) at the given index.

        Args:
            index (int): The index of the data point.

        Returns:
            tuple: A tuple containing the feature vector, pseudo label, and true label.
        """
        index = self.indices[index]
        x = self.features[index]
        y = self.true_labels[index]
        y_hat = self.pseudo_labels[index] if self.pseudo_labels is not None else None
        return x, y_hat, y

    def __len__(self) -> int:
        """
        Gets the number of data points in the dataset.

        Returns:
            int: The number of data points.
        """
        return len(self.indices)
    
    def refine(self, mask: np.ndarray) -> int:
        """
        Refines the dataset by applying a mask to select specific data points.

        Args:
            mask (np.ndarray): A boolean array indicating which data points to keep.
        
        Returns:
            int: The number of data points remaining after applying the mask.
        
        Raises:
            ValueError: If the length of the mask doesn't match the number of data points.
        """
        if len(mask) != len(self.features):
            raise ValueError("Mask length must match the number of data points.")
        
        self.indices = self.indices[mask]
        self.features = self.features[mask]
        self.true_labels = self.true_labels[mask]
        if self.pseudo_labels is not None:
            self.pseudo_labels = self.pseudo_labels[mask]
        if self.pseudo_probabilities is not None:
            self.pseudo_probabilities = self.pseudo_probabilities[mask]
        if self.confidence_scores is not None:
            self.confidence_scores = self.confidence_scores[mask]
        if self.sample_counts is not None:
            self.sample_counts = self.sample_counts[mask]

        return len(self.features)

    def original(self) -> 'MaskedDataset':
        """
        Creates a new MaskedDataset instance with the original dataset without any applied mask.

        Returns:
            MaskedDataset: A new instance of the dataset with the original data.
        """
        return MaskedDataset(self.features, self.true_labels, column_labels=self.column_labels)

    def reset_indices(self) -> None:
        """Resets the indices of the dataset to the original indices."""
        self.indices = self.original_indices.copy()

    def sample(self, N: int, seed: int) -> 'MaskedDataset':
        """
        Samples N data points from the dataset, prioritizing the least sampled points.

        Args:
            N (int): The number of samples to return.
            seed (int): The seed for random number generator.
        
        Returns:
            MaskedDataset: A new instance of the dataset containing N random samples.

        Raises:
            ValueError: If N is greater than the current number of data points in the dataset.
        """
        if N > len(self.features):
            raise ValueError("N cannot be greater than the current number of data points in the dataset.")
        
        # Find the indices of the least sampled points
        sorted_indices = np.argsort(self.sample_counts)
        least_sampled_indices = sorted_indices[:N]
        
        # Set the seed for reproducibility and shuffle the least sampled indices
        np.random.seed(seed)
        np.random.shuffle(least_sampled_indices)
        
        # Select the first N after shuffling
        sampled_indices = least_sampled_indices[:N]
        # Update the sample counts for the sampled indices
        self.sample_counts[sampled_indices] += 1

        # Extract the sampled features and labels
        sampled_features = self.features[sampled_indices, :]
        sampled_true_labels = self.true_labels[sampled_indices]
        sampled_pseudo_labels = self.pseudo_labels[sampled_indices] if self.pseudo_labels is not None else None
        sampled_confidence_scores = self.confidence_scores[sampled_indices] if self.confidence_scores is not None else None
        sampled_pseudo_probs = self.pseudo_probabilities[sampled_indices] if self.pseudo_probabilities is not None else None

        # Return a new MaskedDataset instance containing the sampled data
        sampled_set = MaskedDataset(features=sampled_features, true_labels=sampled_true_labels, column_labels=self.column_labels)
        sampled_set.pseudo_labels = sampled_pseudo_labels
        sampled_set.pseudo_probabilities = sampled_pseudo_probs
        sampled_set.confidence_scores = sampled_confidence_scores
        return sampled_set
    
    def get_features(self) -> np.ndarray:
        """
        Gets the feature vectors of the dataset.

        Returns:
            np.ndarray: The feature vectors of the dataset.
        """
        return self.features
    
    def get_pseudo_labels(self) -> np.ndarray:
        """
        Gets the pseudo labels of the dataset.

        Returns:
            np.ndarray: The pseudo labels of the dataset.
        """
        return self.pseudo_labels
    
    def get_true_labels(self) -> np.ndarray:
        """
        Gets the true labels of the dataset.

        Returns:
            np.ndarray: The true labels of the dataset.
        """
        return self.true_labels
    
    def get_pseudo_probabilities(self) -> np.ndarray:
        """
        Gets the pseudo probabilities of the dataset.

        Returns:
            np.ndarray: The pseudo probabilities of the dataset.
        """
        return self.pseudo_probabilities
    
    def get_confidence_scores(self) -> np.ndarray:
        """
        Gets the confidence scores of the dataset.

        Returns:
            np.ndarray: The confidence scores of the dataset.
        """
        return self.confidence_scores
       
    def set_pseudo_probs_labels(self, pseudo_probabilities: np.ndarray, threshold=0.5) -> None:
        """
        Sets the pseudo probabilities and corresponding pseudo labels for the dataset. The labels are derived by
        applying a threshold to the probabilities.

        Args:
            pseudo_probabilities (np.ndarray): The pseudo probabilities array to be set.
            threshold (float, optional): The threshold to convert probabilities to binary labels. Defaults to 0.5.

        Raises:
            ValueError: If the shape of pseudo_probabilities does not match the number of samples in the features array.
        """
        if pseudo_probabilities.shape[0] != self.features.shape[0]:
            raise ValueError("The shape of pseudo_probabilities must match the number of samples in the features array.")
        
        self.pseudo_probabilities = pseudo_probabilities
        self.pseudo_labels = pseudo_probabilities > threshold
        
    def set_confidence_scores(self, confidence_scores: np.ndarray) -> None:
        """
        Sets the confidence scores for the dataset.

        Args:
            confidence_scores (np.ndarray): The confidence scores array to be set.

        Raises:
            ValueError: If the shape of confidence_scores does not match the number of samples in the features array.
        """
        if confidence_scores.shape[0] != self.features.shape[0]:
            raise ValueError("The shape of confidence_scores must match the number of samples in the features array.")
        
        self.confidence_scores = confidence_scores

    def set_pseudo_labels(self, pseudo_labels: np.ndarray) -> None:
        """
        Adds pseudo labels to the dataset.

        Args:
            pseudo_labels (np.ndarray): The pseudo labels to add.
        
        Raises:
            ValueError: If the length of pseudo_labels does not match the number of samples.
        """
        if len(pseudo_labels) != len(self.features):
            raise ValueError("The length of pseudo_labels must match the number of samples in the dataset.")
        self.pseudo_labels = pseudo_labels

    def clone(self) -> 'MaskedDataset':
        """
        Creates a clone of the current MaskedDataset instance.

        Returns:
            MaskedDataset: A new instance of MaskedDataset containing the same data and configurations as the current instance.
        """
        cloned_set = MaskedDataset(features=self.features.copy(), true_labels=self.true_labels.copy(), column_labels=self.column_labels)
        cloned_set.pseudo_labels = self.pseudo_labels.copy() if self.pseudo_labels is not None else None
        cloned_set.pseudo_probabilities = self.pseudo_probabilities.copy() if self.pseudo_probabilities is not None else None
        cloned_set.confidence_scores = self.confidence_scores.copy() if self.confidence_scores is not None else None

        return cloned_set
    
    def get_info(self) -> dict:
        """
        Returns information about the MaskedDataset.

        Returns:
            dict: A dictionary containing dataset information.
        """
        info = {
            'num_samples': len(self.features),
            'num_features': self.features.shape[1] if self.features.ndim > 1 else 1,
            'has_pseudo_labels': self.pseudo_labels is not None,
            'has_pseudo_probabilities': self.pseudo_probabilities is not None,
            'has_confidence_scores': self.confidence_scores is not None,
        }
        return info
        
    def summary(self) -> None:
        """
        Prints a summary of the dataset.
        """
        info = self.get_info()
        print(f"Number of samples: {info['num_samples']}")
        print(f"Number of features: {info['num_features']}")
        print(f"Has pseudo labels: {info['has_pseudo_labels']}")
        print(f"Has pseudo probabilities: {info['has_pseudo_probabilities']}")
        print(f"Has confidence scores: {info['has_confidence_scores']}")

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the dataset to a pandas DataFrame.

        Returns:
            pd.DataFrame: The dataset as a pandas DataFrame.
        """
        # Convert features to DataFrame
        data = self.features.copy()
        df = pd.DataFrame(data, columns=self.column_labels)
        
        # Add true labels
        df['true_labels'] = self.true_labels
        
        # Add pseudo labels if available
        if self.pseudo_labels is not None:
            df['pseudo_labels'] = self.pseudo_labels
        
        # Add pseudo probabilities if available
        if self.pseudo_probabilities is not None:
            df[f'pseudo_probabilities'] = self.pseudo_probabilities
        
        # Add confidence scores if available
        if self.confidence_scores is not None:
            df['confidence_scores'] = self.confidence_scores
        
        return df

    def save_to_csv(self, file_path: str) -> None:
        """
        Saves the dataset to a CSV file.

        Args:
            file_path (str): The file path to save the dataset to.
        """
        df = self.to_dataframe()
        df.to_csv(file_path, index=False)

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
            ValueError: If an invalid dataset_type is provided or if the shape of features does not match column labels.
        """
        ctx = DataLoadingContext(file)
        column_labels, features_np, true_labels_np = ctx.load_as_np(file, target_column_name)
        
        self.set_column_labels(column_labels)

        # Check if the number of columns in features matches the length of column_labels
        if features_np.shape[1] != len(self.column_labels):
            raise ValueError(f"The shape of features {features_np.shape} does not match the length of column labels {len(column_labels)}")
        
        dataset = MaskedDataset(features_np, true_labels_np, column_labels=self.column_labels)

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
        
    def set_from_data(self, dataset_type: str, features: np.ndarray, true_labels: np.ndarray, column_labels: list = None) -> None:
        """
        Sets the specified dataset using numpy arrays for features and true labels.

        Args:
            dataset_type (str): The type of dataset to set ('training', 'validation', 'reference', 'testing').
            features (np.ndarray): The feature vectors of the dataset.
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

        dataset = MaskedDataset(features, true_labels, column_labels=self.column_labels)
        
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
    
    def summary(self) -> None:
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
            return_instance (bool, optional): If True, returns the MaskedDataset instance; otherwise, returns the features and true labels. Defaults to False.

        Returns:
            Union[tuple, MaskedDataset]: The features and true labels if return_instance is False, otherwise the MaskedDataset instance.

        Raises:
            ValueError: If the base model training set is not initialized.
        """
        if self.base_model_training_set is not None:
            if return_instance:
                return self.base_model_training_set
            return self.base_model_training_set.features, self.base_model_training_set.true_labels
        else:
            raise ValueError("Base model training set not initialized.")

    def __get_base_model_validation_data(self, return_instance: bool = False):
        """
        Retrieves the validation dataset.

        Args:
            return_instance (bool, optional): If True, returns the MaskedDataset instance; otherwise, returns the features and true labels. Defaults to False.

        Returns:
            Union[tuple, MaskedDataset]: The features and true labels if return_instance is False, otherwise the MaskedDataset instance.

        Raises:
            ValueError: If the base model validation set is not initialized.
        """
        if self.base_model_validation_set is not None:
            if return_instance:
                return self.base_model_validation_set
            return self.base_model_validation_set.features, self.base_model_validation_set.true_labels
        else:
            raise ValueError("Base model validation set not initialized.")

    def __get_reference_data(self, return_instance: bool = False):
        """
        Retrieves the reference dataset.

        Args:
            return_instance (bool, optional): If True, returns the MaskedDataset instance; otherwise, returns the features and true labels. Defaults to False.

        Returns:
            Union[tuple, MaskedDataset]: The features and true labels if return_instance is False, otherwise the MaskedDataset instance.

        Raises:
            ValueError: If the reference set is not initialized.
        """
        if self.reference_set is not None:
            if return_instance:
                return self.reference_set
            return self.reference_set.features, self.reference_set.true_labels
        else:
            raise ValueError("Reference set not initialized.")

    def __get_testing_data(self, return_instance: bool = False):
        """
        Retrieves the testing dataset.

        Args:
            return_instance (bool, optional): If True, returns the MaskedDataset instance; otherwise, returns the features and true labels. Defaults to False.

        Returns:
            Union[tuple, MaskedDataset]: The features and true labels if return_instance is False, otherwise the MaskedDataset instance.

        Raises:
            ValueError: If the testing set is not initialized.
        """
        if self.testing_set is not None:
            if return_instance:
                return self.testing_set
            return self.testing_set.features, self.testing_set.true_labels
        else:
            raise ValueError("Testing set not initialized.")
