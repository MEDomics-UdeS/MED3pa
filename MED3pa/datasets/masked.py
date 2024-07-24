"""
The masked.py module includes the ``MaskedDataset`` class that is capable of handling many dataset related operations, such as cloning, sampling, refining...etc.
"""

import numpy as np
import pandas as pd

from torch.utils.data import Dataset


class MaskedDataset(Dataset):
    """
    A dataset wrapper for PyTorch that supports masking and sampling of data points.
    """
    
    def __init__(self, observations: np.ndarray, true_labels: np.ndarray, column_labels: list = None):
        """
        Initializes the MaskedDataset.

        Args:
            observations (np.ndarray): The observations vectors of the dataset.
            true_labels (np.ndarray): The true labels of the dataset.
            column_labels (list, optional): The column labels for the observation vectors. Defaults to None.
        """
        self.__observations = observations
        self.__true_labels = true_labels
        self.__indices = np.arange(len(self.__observations))
        self.__original_indices = self.__indices.copy()
        self.__sample_counts = np.zeros(len(observations), dtype=int)
        self.__pseudo_probabilities = None
        self.__pseudo_labels = None 
        self.__confidence_scores = None
        self.__column_labels = column_labels if column_labels is not None else [f'feature_{i}' for i in range(observations.shape[1])]
        self.__file_path = None

    def __getitem__(self, index: int) -> tuple:
        """
        Retrieves the data point and its label(s) at the given index.

        Args:
            index (int): The index of the data point.

        Returns:
            tuple: A tuple containing the observation vector, pseudo label, and true label.
        """
        index = self.__indices[index]
        x = self.__observations[index]
        y = self.__true_labels[index]
        y_hat = self.__pseudo_labels[index] if self.__pseudo_labels is not None else None
        return x, y_hat, y

    def __len__(self) -> int:
        """
        Gets the number of data points in the dataset.

        Returns:
            int: The number of data points.
        """
        return len(self.__indices)
    
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
        if len(mask) != len(self.__observations):
            raise ValueError("Mask length must match the number of data points.")
        
        self.__indices = self.__indices[mask]
        self.__observations = self.__observations[mask]
        self.__true_labels = self.__true_labels[mask]
        if self.__pseudo_labels is not None:
            self.__pseudo_labels = self.__pseudo_labels[mask]
        if self.__pseudo_probabilities is not None:
            self.__pseudo_probabilities = self.__pseudo_probabilities[mask]
        if self.__confidence_scores is not None:
            self.__confidence_scores = self.__confidence_scores[mask]
        if self.__sample_counts is not None:
            self.__sample_counts = self.__sample_counts[mask]

        return len(self.__observations)

    def reset_indices(self) -> None:
        """Resets the indices of the dataset to the original indices."""
        self.__indices = self.__original_indices.copy()

    def sample_uniform(self, N: int, seed: int) -> 'MaskedDataset':
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
        if N > len(self.__observations):
            raise ValueError("N cannot be greater than the current number of data points in the dataset.")
        
        # Find the indices of the least sampled points
        sorted_indices = np.argsort(self.__sample_counts)
        least_sampled_indices = sorted_indices[:N]
        
        # Set the seed for reproducibility and shuffle the least sampled indices
        np.random.seed(seed)
        np.random.shuffle(least_sampled_indices)
        
        # Select the first N after shuffling
        sampled_indices = least_sampled_indices[:N]
        # Update the sample counts for the sampled indices
        self.__sample_counts[sampled_indices] += 1

        # Extract the sampled observations and labels
        sampled_obs = self.__observations[sampled_indices, :]
        sampled_true_labels = self.__true_labels[sampled_indices]
        sampled_pseudo_labels = self.__pseudo_labels[sampled_indices] if self.__pseudo_labels is not None else None
        sampled_confidence_scores = self.__confidence_scores[sampled_indices] if self.__confidence_scores is not None else None
        sampled_pseudo_probs = self.__pseudo_probabilities[sampled_indices] if self.__pseudo_probabilities is not None else None

        # Return a new MaskedDataset instance containing the sampled data
        sampled_set = MaskedDataset(observations=sampled_obs, true_labels=sampled_true_labels, column_labels=self.__column_labels)
        sampled_set.set_pseudo_probs_labels(sampled_pseudo_probs) if sampled_pseudo_probs is not None else None
        sampled_set.set_pseudo_labels(sampled_pseudo_labels) if sampled_pseudo_labels is not None else None
        sampled_set.set_confidence_scores(sampled_confidence_scores) if sampled_confidence_scores is not None else None
        return sampled_set
    
    def sample_random(self, N: int, seed: int) -> 'MaskedDataset':
        """
        Samples N data points randomly from the dataset using the given seed.

        Args:
            N (int): The number of samples to return.
            seed (int): The seed for random number generator.

        Returns:
            MaskedDataset: A new instance of the dataset containing N random samples.

        Raises:
            ValueError: If N is greater than the current number of data points in the dataset.
        """
        if N > len(self.__observations):
            raise ValueError("N cannot be greater than the current number of data points in the dataset.")

        # Set the seed for reproducibility and generate random indices
        rng = np.random.RandomState(seed)
        random_indices = rng.permutation(len(self.__observations))[:N]

        # Extract the sampled observations and labels
        sampled_obs = self.__observations[random_indices, :]
        sampled_true_labels = self.__true_labels[random_indices]
        sampled_pseudo_labels = self.__pseudo_labels[random_indices] if self.__pseudo_labels is not None else None
        sampled_confidence_scores = self.__confidence_scores[random_indices] if self.__confidence_scores is not None else None
        sampled_pseudo_probs = self.__pseudo_probabilities[random_indices] if self.__pseudo_probabilities is not None else None

        # Return a new MaskedDataset instance containing the sampled data
        sampled_set = MaskedDataset(observations=sampled_obs, true_labels=sampled_true_labels, column_labels=self.__column_labels)
        sampled_set.set_pseudo_probs_labels(sampled_pseudo_probs) if sampled_pseudo_probs is not None else None
        sampled_set.set_pseudo_labels(sampled_pseudo_labels) if sampled_pseudo_labels is not None else None
        sampled_set.set_confidence_scores(sampled_confidence_scores) if sampled_confidence_scores is not None else None
        return sampled_set
    
    def get_observations(self) -> np.ndarray:
        """
        Gets the observations vectors of the dataset.

        Returns:
            np.ndarray: The observations vectors of the dataset.
        """
        return self.__observations
    
    def get_pseudo_labels(self) -> np.ndarray:
        """
        Gets the pseudo labels of the dataset.

        Returns:
            np.ndarray: The pseudo labels of the dataset.
        """
        return self.__pseudo_labels
    
    def get_true_labels(self) -> np.ndarray:
        """
        Gets the true labels of the dataset.

        Returns:
            np.ndarray: The true labels of the dataset.
        """
        return self.__true_labels
    
    def get_pseudo_probabilities(self) -> np.ndarray:
        """
        Gets the pseudo probabilities of the dataset.

        Returns:
            np.ndarray: The pseudo probabilities of the dataset.
        """
        return self.__pseudo_probabilities
    
    def get_confidence_scores(self) -> np.ndarray:
        """
        Gets the confidence scores of the dataset.

        Returns:
            np.ndarray: The confidence scores of the dataset.
        """
        return self.__confidence_scores
    
    def get_sample_counts(self) -> np.ndarray:
        """
        Gets the how many times each element of the dataset was sampled.

        Returns:
            np.ndarray: The sample counts of the dataset.
        """
        return self.__sample_counts
    
    def get_file_path(self) -> str :
        """
        Gets the file path of the dataset if it has been set from a file.

        Returns:
            str: The file path of the dataset.
        """
        return self.__file_path
    
    def set_pseudo_probs_labels(self, pseudo_probabilities: np.ndarray, threshold=0.5) -> None:
        """
        Sets the pseudo probabilities and corresponding pseudo labels for the dataset. The labels are derived by
        applying a threshold to the probabilities.

        Args:
            pseudo_probabilities (np.ndarray): The pseudo probabilities array to be set.
            threshold (float, optional): The threshold to convert probabilities to binary labels. Defaults to 0.5.

        Raises:
            ValueError: If the shape of pseudo_probabilities does not match the number of samples in the observations array.
        """
        if pseudo_probabilities.shape[0] != self.__observations.shape[0]:
            raise ValueError("The shape of pseudo_probabilities must match the number of samples in the observations array.")
        
        self.__pseudo_probabilities = pseudo_probabilities
        self.__pseudo_labels = pseudo_probabilities >= threshold
        
    def set_confidence_scores(self, confidence_scores: np.ndarray) -> None:
        """
        Sets the confidence scores for the dataset.

        Args:
            confidence_scores (np.ndarray): The confidence scores array to be set.

        Raises:
            ValueError: If the shape of confidence_scores does not match the number of samples in the observations array.
        """
        if confidence_scores.shape[0] != self.__observations.shape[0]:
            raise ValueError("The shape of confidence_scores must match the number of samples in the observations array.")
        
        self.__confidence_scores = confidence_scores

    def set_pseudo_labels(self, pseudo_labels: np.ndarray) -> None:
        """
        Adds pseudo labels to the dataset.

        Args:
            pseudo_labels (np.ndarray): The pseudo labels to add.
        
        Raises:
            ValueError: If the length of pseudo_labels does not match the number of samples.
        """
        if len(pseudo_labels) != len(self.__observations):
            raise ValueError("The length of pseudo_labels must match the number of samples in the dataset.")
        self.__pseudo_labels = pseudo_labels

    def set_file_path(self, file: str) -> None:
        """
        Sets the file path of the dataset if it has been set from a file.

        Args:
            file (str): The file path of the dataset.

        """
        self.__file_path = file
        
    def clone(self) -> 'MaskedDataset':
        """
        Creates a clone of the current MaskedDataset instance.

        Returns:
            MaskedDataset: A new instance of MaskedDataset containing the same data and configurations as the current instance.
        """
        cloned_set = MaskedDataset(observations=self.__observations.copy(), true_labels=self.__true_labels.copy(), column_labels=self.__column_labels)
        cloned_set.__pseudo_labels = self.__pseudo_labels.copy() if self.__pseudo_labels is not None else None
        cloned_set.__pseudo_probabilities = self.__pseudo_probabilities.copy() if self.__pseudo_probabilities is not None else None
        cloned_set.__confidence_scores = self.__confidence_scores.copy() if self.__confidence_scores is not None else None

        return cloned_set
    
    def get_info(self) -> dict:
        """
        Returns information about the MaskedDataset.

        Returns:
            dict: A dictionary containing dataset information.
        """
        info = {
            'file_path': self.__file_path,
            'num_samples': len(self.__observations),
            'num_observations': self.__observations.shape[1] if self.__observations.ndim > 1 else 1,
            'has_pseudo_labels': self.__pseudo_labels is not None,
            'has_pseudo_probabilities': self.__pseudo_probabilities is not None,
            'has_confidence_scores': self.__confidence_scores is not None,
        }
        return info
        
    def summarize(self) -> None:
        """
        Prints a summary of the dataset.
        """
        info = self.get_info()
        print(f"Number of samples: {info['num_samples']}")
        print(f"Number of observations: {info['num_observations']}")
        print(f"Has pseudo labels: {info['has_pseudo_labels']}")
        print(f"Has pseudo probabilities: {info['has_pseudo_probabilities']}")
        print(f"Has confidence scores: {info['has_confidence_scores']}")

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the dataset to a pandas DataFrame.

        Returns:
            pd.DataFrame: The dataset as a pandas DataFrame.
        """
        # Convert observations to DataFrame
        data = self.__observations.copy()
        df = pd.DataFrame(data, columns=self.__column_labels)
        
        # Add true labels
        df['true_labels'] = self.__true_labels
        
        # Add pseudo labels if available
        if self.__pseudo_labels is not None:
            df['pseudo_labels'] = self.__pseudo_labels
        
        # Add pseudo probabilities if available
        if self.__pseudo_probabilities is not None:
            df[f'pseudo_probabilities'] = self.__pseudo_probabilities
        
        # Add confidence scores if available
        if self.__confidence_scores is not None:
            df['confidence_scores'] = self.__confidence_scores
        
        return df

    def save_to_csv(self, file_path: str) -> None:
        """
        Saves the dataset to a CSV file.

        Args:
            file_path (str): The file path to save the dataset to.
        """
        df = self.to_dataframe()
        df.to_csv(file_path, index=False)

    def combine(self, other: 'MaskedDataset') -> 'MaskedDataset':
        """
        Combines the current MaskedDataset with another MaskedDataset.

        Args:
            other (MaskedDataset): The other MaskedDataset to combine with.
        
        Returns:
            MaskedDataset: A new instance of MaskedDataset containing the combined data.

        Raises:
            ValueError: If the column labels of the two datasets do not match.
        """
        if self.__column_labels != other.__column_labels:
            raise ValueError("The column labels of the two datasets must match to combine them.")

        combined_observations = np.vstack((self.__observations, other.__observations))
        combined_true_labels = np.concatenate((self.__true_labels, other.__true_labels))
        combined_pseudo_labels = np.concatenate((self.__pseudo_labels, other.__pseudo_labels)) if self.__pseudo_labels is not None and other.__pseudo_labels is not None else None
        combined_pseudo_probabilities = np.concatenate((self.__pseudo_probabilities, other.__pseudo_probabilities)) if self.__pseudo_probabilities is not None and other.__pseudo_probabilities is not None else None
        combined_confidence_scores = np.concatenate((self.__confidence_scores, other.__confidence_scores)) if self.__confidence_scores is not None and other.__confidence_scores is not None else None

        combined_dataset = MaskedDataset(
            observations=combined_observations,
            true_labels=combined_true_labels,
            column_labels=self.__column_labels
        )
        combined_dataset.set_pseudo_labels(combined_pseudo_labels) if combined_pseudo_labels is not None else None
        combined_dataset.set_pseudo_probs_labels(combined_pseudo_probabilities) if combined_pseudo_probabilities is not None else None
        combined_dataset.set_confidence_scores(combined_confidence_scores) if combined_confidence_scores is not None else None

        return combined_dataset
