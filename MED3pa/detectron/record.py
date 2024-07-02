"""
This module is crucial for tracking and managing the results across multiple runs of the Detectron method. 
It defines the DetectronRecord class, which captures individual records of Detectron results, storing evaluation metrics and probabilities associated with each model in the ensemble and across each run. 
The DetectronRecordsManager class manages a collection of these records, facilitating updates, retrieval, and analysis of the results.
"""
import numpy as np
import pandas as pd

from MED3pa.models.abstract_models import Model

class DetectronRecord:
    """
    Represents a single record of Detectron results, storing metrics and probabilities associated with a model's evaluation after one run.
    the model with id 0 represents the base model, whilst other ids represent the cdcs present in the ensemble.
    """
    
    def __init__(self, seed: int, model_id: int, original_count: int) -> None:
        """
        Initializes a DetectronRecord with the specified seed, model identifier, and initial sample count.

        Args:
            seed (int): The seed value used for the model run.
            model_id (int): Identifier for the specific model within the ensemble.
            original_count (int): The initial count of samples before any point was rejected.
        """
        self.seed = seed
        self.model_id = model_id
        self.original_count = original_count

    def update(self, validation_auc: float, test_auc: float, predicted_probabilities: np.ndarray, count: int):
        """
        Updates the record with evaluation metrics and the count of samples after rejection.

        Args:
            validation_auc (float): The area under the curve (AUC) value on the validation set.
            test_auc (float): The area under the curve (AUC) value on the test set.
            predicted_probabilities (np.ndarray): Predicted probabilities from the model for the test set.
            count (int): The number of samples remaining after applying the rejection criterion.
        """
        self.validation_auc = validation_auc
        self.test_auc = test_auc
        self.predicted_probabilities = predicted_probabilities
        self.updated_count = count
        self.rejected_samples = self.original_count - self.updated_count
    
    def to_dict(self) -> dict:
        """
        Converts the DetectronRecord into a dictionary for easier serialization and manipulation.

        Returns:
            dict: A dictionary representation of the DetectronRecord with all attributes.
        """
        return {
            'seed': self.seed,
            'model_id': self.model_id,
            'validation_auc': self.validation_auc,
            'test_auc': self.test_auc,
            'rejection_rate': 1 - self.updated_count / self.original_count,
            'predicted_probabilities': self.predicted_probabilities.tolist(),
            'count': self.updated_count,
            'rejected_count': self.rejected_samples
        }


class DetectronRecordsManager:
    """
    Manages a collection of DetectronRecords, providing methods to update, retrieve, and analyze the records.
    """
    def __init__(self, sample_size : int):
        """
        Initializes the DetectronRecordsManager with a specified sample size.

        Args:
            sample_size (int): The size of samples used in Detectron experiments, used to initialize records.
        """
        self.records = []
        self.sample_size = sample_size
        self.idx = 0
        self.__seed = None
        self.sampling_counts = None

    def seed(self, seed: int):
        """
        Sets the seed used for updating records, ensuring consistency and reproducibility in experiments.

        Args:
            seed (int): The seed value to set for this set of updates.
        """
        self.__seed = seed


    def update(self, val_data_x: np.ndarray, val_data_y: np.ndarray, 
               sample_size: int, model: Model, model_id: int,
               predicted_probabilities: np.ndarray=None, test_data_x: np.ndarray=None, test_data_y: np.ndarray=None):
        """
        Updates the records manager with new run results, adding a new DetectronRecord.

        Args:
            val_data_x (np.ndarray): observations from the validation dataset used for evaluation.
            val_data_y (np.ndarray): True labels from the validation dataset.
            sample_size (int): The number of samples used in this update.
            model (Model): The model instance used for evaluation, which should have an `evaluate` method.
            model_id (int): The identifier of the model within the ensemble.
            predicted_probabilities (np.ndarray, optional): Predicted probabilities from the model on the test dataset.
            test_data_x (np.ndarray, optional): observations from the test dataset used for evaluation.
            test_data_y (np.ndarray, optional): True labels from the test dataset.
        """
        assert self.__seed is not None, 'Seed must be set before updating the record'
        
        record = DetectronRecord(self.__seed, model_id, self.sample_size)
        validation_auc = model.evaluate(val_data_x, val_data_y, ['Auc']).get('Auc')
        testing_auc = model.evaluate(test_data_x, test_data_y, ['Auc']).get('Auc') if test_data_x is not None else float('nan')
        record.update(validation_auc, testing_auc, predicted_probabilities, sample_size)
        self.records.append(record.to_dict())
        self.idx += 1

    def freeze(self):
        """
        Finalizes the records, converting them into a pandas DataFrame for easier manipulation and analysis.
        """
        self.records = self.get_record()

    def get_record(self):
        """
        Retrieves the current records as a pandas DataFrame.

        Returns:
            pd.DataFrame: The records stored in the manager, formatted as a DataFrame.
        """
        if isinstance(self.records, pd.DataFrame):
            return self.records
        else:
            return pd.DataFrame(self.records)

    def save(self, path:str):
        """
        Saves the current records to a CSV file at the specified path.

        Args:
            path (str): The file path where the records DataFrame will be saved.
        """
        self.get_record().to_csv(path, index=False)

    @staticmethod
    def load(path:str):
        """
        Loads records from a CSV file into a DetectronRecordsManager instance.

        Args:
            path (str): The file path from which to load the records.

        Returns:
            DetectronRecordsManager: A new instance of DetectronRecordsManager containing the loaded records.
        """
        x = DetectronRecordsManager(sample_size=None)
        x.records = pd.read_csv(path)
        x.sample_size = x.records.query('model_id==0').iloc[0]['count']
        return x

    def counts(self, max_ensemble_size:int=None) -> np.ndarray:
        """
        Retrieves the number of samples kept for each run, optionally limited to a maximum ensemble size.

        Args:
            max_ensemble_size (int, optional): The maximum size of the ensemble to consider for counts.

        Returns:
            np.ndarray: An array containing the kept data points count after each Detectron run.
        """
        assert max_ensemble_size is None or max_ensemble_size > 0, 'max_ensemble_size must be positive or None'
        rec = self.get_record()
        counts = []
        for i in rec.seed.unique():
            run = rec.query(f'seed=={i}')
            if max_ensemble_size is not None:
                run = run.iloc[:max_ensemble_size + 1]
            counts.append(run.iloc[-1]['count'])
        return np.array(counts)
    
    def rejection_rates(self, max_ensemble_size:int=None) -> np.ndarray:
        """
        Calculates the rejection rates for each run, optionally limited to a certain number of models.

        Args:
            max_ensemble_size (int, optional): The maximum number of models to consider for calculating rejection rates.

        Returns:
            np.ndarray: An array containing the rejection rates after each Detectron run.
        """
        assert max_ensemble_size is None or max_ensemble_size > 0, 'max_ensemble_size must be positive or None'
        rec = self.get_record()
        counts = []
        for i in rec.seed.unique():
            run = rec.query(f'seed=={i}')
            if max_ensemble_size is not None:
                run = run.iloc[:max_ensemble_size + 1]
            counts.append(run.iloc[-1]['rejection_rate'])
        return np.array(counts)
    
    def predicted_probabilities(self, max_ensemble_size:int=None):
        """
        Retrieves the predicted probabilities for each model in the ensemble for each run.

        Args:
            max_ensemble_size (int, optional): The maximum number of models to consider for collecting predicted probabilities.

        Returns:
            np.ndarray: An array of predicted probabilities for each model in the ensemble for each run.
        """
        assert max_ensemble_size is None or max_ensemble_size > 0, 'max_ensemble_size must be positive or None'
        rec = self.get_record()
        seed_probs = []

        for i in rec.seed.unique():
            run = rec.query(f'seed=={i}')
            if max_ensemble_size is not None:
                run = run.iloc[:max_ensemble_size + 1]
            probs = []
            base_model_probs = None
            cdcs_probs = []
            for _, row in run.iterrows():
                if row['model_id'] == 0:
                    base_model_probs = row['predicted_probabilities']
                else:
                    cdcs_probs.append(row['predicted_probabilities'])
                probs.append(row['predicted_probabilities'])
            averaged_cdcs_probs = np.mean(cdcs_probs, axis=0)
            seed_probs.append(np.array([base_model_probs, averaged_cdcs_probs]))

        # Convert the list of arrays to a 2D NumPy array
        predicted_probs_array = np.array(seed_probs)

        # Check for consistency in the shape of predicted probabilities
        if not all(len(prob) == len(seed_probs[0][0]) for seed in seed_probs for prob in seed):
            raise ValueError("Inconsistent shapes found in predicted probabilities")

        return predicted_probs_array
    
    def rejected_counts(self, max_ensemble_size:int=None) -> np.ndarray:
        """
        Retrieves the number of samples rejected for each run, optionally limited to a maximum ensemble size.

        Args:
            max_ensemble_size (int, optional): The maximum size of the ensemble to consider for counts.

        Returns:
            np.ndarray: An array containing the rejected data points count after each Detectron run.
        """
        assert max_ensemble_size is None or max_ensemble_size > 0, 'max_ensemble_size must be positive or None'
        rec = self.get_record()
        counts = []
        for i in rec.seed.unique():
            run = rec.query(f'seed=={i}')
            if max_ensemble_size is not None:
                run = run.iloc[:max_ensemble_size + 1]
            counts.append(run.iloc[-1]['rejected_count'])
        return np.array(counts)

    def count_quantile(self, quantile, max_ensemble_size:int=None):
        """
        return the specified quantile of the kept points counts
        """
        counts = self.counts(max_ensemble_size)
        return np.quantile(counts, quantile, method='inverted_cdf')
    
    def rejected_count_quantile(self, quantile, max_ensemble_size:int=None):
        """
        return the specified quantile of the rejected points counts
        """
        rejected_counts = self.rejected_counts(max_ensemble_size=max_ensemble_size)
        return np.quantile(rejected_counts, quantile)


