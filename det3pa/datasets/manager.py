"""
This module provides classes for managing datasets.
"""

import numpy as np
from torch.utils.data import Dataset
from .loading_context import DataLoadingContext

class MaskedDataset(Dataset):
    """
    A dataset wrapper for PyTorch that supports masking and sampling of data points.
    
    Attributes:
        features (np.ndarray): The feature vectors of the dataset.
        true_labels (np.ndarray): The true labels of the dataset.
        pseudo_labels (np.ndarray, optional): The pseudo labels predicted by a model.
        pseudo_probabilities (np.ndarray, optional): The pseudo probabilities associated with the pseudo labels.
        indices (np.ndarray): The current indices of the dataset after applying masking.
        original_indices (np.ndarray): The original indices of the dataset.
        sample_counts (np.ndarray): Tracks the number of times each point is sampled, after using the sample() method.
    """
    
    def __init__(self, features: np.ndarray, true_labels: np.ndarray, mask=True, pseudo_labels: np.ndarray = None, pseudo_probabilities: np.ndarray = None):
        """
        Initializes the MaskedDataset.

        Args:
            features (np.ndarray): The feature vectors of the dataset.
            true_labels (np.ndarray): The true labels of the dataset.
            mask (bool, optional): Indicates if the dataset is masked. Defaults to True.
            pseudo_labels (np.ndarray, optional): The pseudo labels for masked data points. Defaults to None.
            pseudo_probabilities (np.ndarray, optional): The pseudo probabilities associated with the pseudo labels. Defaults to None.
        """
        self.features = features
        self.true_labels = true_labels
        self.pseudo_labels = pseudo_labels
        self.pseudo_probabilities = pseudo_probabilities
        self.indices = np.arange(len(self.features))
        self.original_indices = self.indices.copy()
        self.sample_counts = np.zeros(len(features), dtype=int)  

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
        
        return len(self.features)

    def original(self) -> 'MaskedDataset':
        """
        Creates a new MaskedDataset instance with the original dataset without any applied mask.

        Returns:
            MaskedDataset: A new instance of the dataset with the original data.
        """
        return MaskedDataset(self.features, self.true_labels, mask=False, pseudo_labels=self.pseudo_labels)

    def reset_indices(self) -> None:
        """Resets the indices of the dataset to the original indices."""
        self.indices = self.original_indices.copy()

    def __len__(self) -> int:
        """
        Gets the number of data points in the dataset.

        Returns:
            int: The number of data points.
        """
        return len(self.indices)
    
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
        least_sampled_indices = sorted_indices[:N]  # Take more than needed to add randomness
        
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

        # Return a new MaskedDataset instance containing the sampled data
        return MaskedDataset(sampled_features, sampled_true_labels, sampled_pseudo_labels)
    
    def get_features(self) -> np.ndarray:
        """
        Gets the pseudo labels of the dataset.

        Returns:
            np.ndarray: The pseudo labels of the dataset.
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
        
    def clone(self) -> 'MaskedDataset':
        """
        Creates a clone of the current MaskedDataset instance.

        Returns:
            MaskedDataset: A new instance of MaskedDataset containing the same data and configurations as the current instance.
        """
        return MaskedDataset(
            features=self.features.copy(),
            true_labels=self.true_labels.copy(),
            pseudo_labels=self.pseudo_labels.copy() if self.pseudo_labels is not None else None,
            pseudo_probabilities=self.pseudo_probabilities.copy() if self.pseudo_probabilities is not None else None
        )

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
    
    def set_base_model_training_data(self, training_file: str, target_column_name: str) -> None:
        """
        Loads and sets the base model training dataset from a file.

        Args:
            training_file (str): The file path to the training data.
            target_column_name (str): The name of the target column in the dataset.
        """
        ctx = DataLoadingContext(training_file)
        column_labels, features_np, true_labels_np = ctx.load_as_np(training_file, target_column_name)
        self.base_model_training_set = MaskedDataset(features_np, true_labels_np)
        if self.column_labels is None:
            self.column_labels = column_labels
        else:
            if self.column_labels != column_labels:
                raise ValueError("Column labels extracted from the training dataset do not match the existing column labels.")

    
    def set_base_model_validation_data(self, validation_file: str, target_column_name: str) -> None:
        """
        Loads and sets the base model validation dataset from a file.

        Args:
            validation_file (str): The file path to the validation data.
            target_column_name (str): The name of the target column in the dataset.
        """
        ctx = DataLoadingContext(validation_file)
        column_labels, features_np, true_labels_np = ctx.load_as_np(validation_file, target_column_name)
        self.base_model_validation_set = MaskedDataset(features_np, true_labels_np)
        if self.column_labels is None:
            self.column_labels = column_labels
        else:
            if self.column_labels != column_labels:
                raise ValueError("Column labels extracted from the validation dataset do not match the existing column labels.")

    def set_reference_data(self, reference_file: str, target_column_name: str) -> None:
        """
        Loads and sets the reference dataset from a file.

        Args:
            reference_file (str): The file path to the reference data.
            target_column_name (str): The name of the target column in the dataset.
        """
        ctx = DataLoadingContext(reference_file)
        column_labels, features_np, true_labels_np = ctx.load_as_np(reference_file, target_column_name)
        self.reference_set = MaskedDataset(features_np, true_labels_np)
        if self.column_labels is None:
            self.column_labels = column_labels
        else:
            if self.column_labels != column_labels:
                raise ValueError("Column labels extracted from the reference dataset do not match the existing column labels.")
        
    def set_testing_data(self, testing_file: str, target_column_name: str) -> None:
        """
        Loads and sets the testing dataset from a file.

        Args:
            testing_file (str): The file path to the testing data.
            target_column_name (str): The name of the target column in the dataset.
        """
        ctx = DataLoadingContext(testing_file)
        column_labels, features_np, true_labels_np = ctx.load_as_np(testing_file, target_column_name)
        self.testing_set = MaskedDataset(features_np, true_labels_np)
        if self.column_labels is None:
            self.column_labels = column_labels
        else:
            if self.column_labels != column_labels:
                raise ValueError("Column labels extracted from the testing dataset do not match the existing column labels.")

    def get_base_model_training_data(self, return_instance: bool = False):
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

    def get_base_model_validation_data(self, return_instance: bool = False):
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

    def get_reference_data(self, return_instance: bool = False):
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

    def get_testing_data(self, return_instance: bool = False):
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
