"""
This module houses the ``DetectronEnsemble`` class, responsible for managing the Constrained Disagreement Classifiers (CDCs) ensemble. 
It coordinates the training and evaluation of multiple CDCs, aiming to disagree with the predictions of a primary base model under specified conditions.
The ensemble leverages a base model, provided by ``BaseModelManager``, to generate models that are designed to systematically disagree with it in a controlled fashion.
"""
import numpy as np

from tqdm import tqdm

from MED3pa.datasets import DatasetsManager
from MED3pa.models.base import BaseModelManager
from .record import DetectronRecordsManager
from .stopper import EarlyStopper


class DetectronEnsemble:
    """
    Manages the constrained disagreement classifiers (CDCs) ensemble, designed to disagree with the base model
    under specific conditions. This class facilitates the training and evaluation of multiple CDCs, with a focus
    on generating models that systematically challenge the predictions of a primary base model.
    """
    def __init__(self, base_model_manager: BaseModelManager, ens_size):
        """
        Initializes the Detectron Ensemble with a specified base model manager and ensemble size.

        Args:
            base_model_manager (BaseModelManager): The manager for handling the base model operations, responsible
                                                   for training, prediction, and general management of the base model.
            ens_size (int): The number of CDCs in the ensemble. This does not include the base model itself.

        Attributes:
            base_model_manager (BaseModelManager): Instance of BaseModelManager that manages the operations of the base model.
            base_model (Model): The actual base model instance retrieved from the model manager.
            ens_size (int): Number of CDC models in the ensemble.
            cdcs (list of Model): List containing clones of the base model, which are used as CDCs in the ensemble.
        """
        self.base_model_manager = base_model_manager
        self.base_model = base_model_manager.get_instance()
        self.ens_size = ens_size
        self.cdcs = [base_model_manager.clone_base_model() for _ in range(ens_size)]

    def evaluate_ensemble(self, 
                          datasets : DatasetsManager, 
                          n_runs : int, 
                          samples_size : int , 
                          training_params : dict, 
                          set : str = 'reference', 
                          patience : int = 3, 
                          allow_margin : bool = False,
                          margin : int = None):
        
        """
        Trains the CDCs ensemble to disagree with the base model on a subset of data present in datasets. This process 
        is repeated for a specified number of runs, each using a different sample of the data.

        Args:
            datasets (DatasetsManager): Holds the datasets used for training and validation of the base model, as well as the 
                                        reference and testing sets for the Detectron.
            n_runs (int): Number of runs to train the ensemble. Each run uses a new random sample of data points.
            sample_size (int): Number of points to use in each run.
            training_params (dict): Additional parameters to use for training the ensemble models.
            set (str, optional): Specifies the dataset used for training the ensemble. Options are 'reference' or 'testing'.
                                 Default is 'reference'.
            patience (int, optional): The number of consecutive updates without improvement to wait before early stopping.
                                      Default is 3.
            allow_margin (bool, optional): Whether to use a probability margin to refine the disagreement. Default is False.
            margin (float, optional): The margin threshold above which disagreements in probabilities between the base model 
                                      and ensemble are considered significant, if allow_margin is True.

        Returns:
            DetectronRecordsManager: The records manager containing all the evaluation records from the ensemble runs.

        Raises:
            ValueError: If the specified set is neither 'reference' nor 'testing'.
        """
        # set up the training, validation and testing sets
        training_data = datasets.get_dataset_by_type(dataset_type="training", return_instance=True)
        validation_data = datasets.get_dataset_by_type(dataset_type="validation", return_instance=True)
        if set=='reference':
            testing_data = datasets.get_dataset_by_type(dataset_type="reference", return_instance=True)
        elif set == 'testing':
            testing_data = datasets.get_dataset_by_type(dataset_type="testing", return_instance=True)
        else:
            raise ValueError("The set used to evaluate the ensemble must be either the reference set or the testing set")

        # set up the records manager
        record = DetectronRecordsManager(sample_size=samples_size)

        # evaluate the ensemble for n_runs of runs
        for seed in tqdm(range(n_runs), desc='running seeds'):
            # sample the testing set according to the provided sample_size and current seed
            testing_set = testing_data.sample_uniform(samples_size, seed)

            # predict probabilities using the base model on the testing set
            base_model_pred_probs = self.base_model.predict(testing_set.get_observations(), True)

            # set pseudo probabilities and pseudo labels predicted by the base model
            testing_set.set_pseudo_probs_labels(base_model_pred_probs, 0.5)
            cloned_testing_set = testing_set.clone()

            # the base model is always the model with id = 0
            model_id = 0

            # seed the record
            record.seed(seed)

            # update the record with the results of the base model
            record.update(val_data_x=validation_data.get_observations(), val_data_y=validation_data.get_true_labels(), 
                          sample_size=samples_size, model=self.base_model, model_id=model_id, 
                          predicted_probabilities=testing_set.get_pseudo_probabilities(), 
                          test_data_x=testing_set.get_observations(), test_data_y=testing_set.get_true_labels())
            
            # set up the Early stopper
            stopper = EarlyStopper(patience=patience, mode='min')
            stopper.update(samples_size)

            # Initialize the updated count
            updated_count = samples_size

            # Train the cdcs
            for i in range(1, self.ens_size + 1):
                # get the current cdc
                cdc = self.cdcs[i-1]
                
                # save the model id
                model_id = i
                
                # update the training params with the current seed which is the model id
                if training_params is not None :
                    training_params.update({'seed':i})
                else:
                    training_params = {'seed':i}

                # train this cdc to disagree
                cdc.train_to_disagree(x_train=training_data.get_observations(), y_train=training_data.get_true_labels(), 
                                      x_validation=validation_data.get_observations(), y_validation=validation_data.get_true_labels(), 
                                      x_test=testing_set.get_observations(), y_test=testing_set.get_pseudo_labels(),
                                      training_parameters=training_params,
                                      balance_train_classes=True, 
                                      N=updated_count)
                
                # predict probabilities using this cdc
                cdc_probabilities = cdc.predict(testing_set.get_observations(), True)
                cdc_probabilities_original_set = cdc.predict(cloned_testing_set.get_observations(), True)

                # deduct the predictions of this cdc
                cdc_predicitons = cdc_probabilities >= 0.5

                # calculate the mask to refine the testing set
                mask = (cdc_predicitons == testing_set.get_pseudo_labels())

                # If margin is specified and there are disagreements, check if the probabilities are significatly different
                if allow_margin and not np.all(mask):

                    # convert to disagreement mask
                    disagree_mask = ~mask
                    
                    # calculate the difference between cdc probs and bm probs
                    prob_diff = np.abs(testing_set.get_pseudo_probabilities() - cdc_probabilities)
                    
                    # in the disagreement mask, keep only the data point where the probability difference is greater than the margin, only for disagreed on points
                    refine_mask = (prob_diff < margin) & disagree_mask

                    # update the mask according to the refine_mask array
                    mask[refine_mask] = True
                
                # refine the testing set using the mask                
                updated_count = testing_set.refine(mask)

                # log the results for this model
                record.update(val_data_x=validation_data.get_observations(), val_data_y=validation_data.get_true_labels(),
                              sample_size=updated_count, predicted_probabilities=cdc_probabilities_original_set, 
                              model=cdc, model_id=model_id)
                
                # break if no more data
                if updated_count == 0:
                    break

                if stopper.update(updated_count):
                    # print(f'Early stopping: Converged after {i} models')
                    break
        
        record.sampling_counts = testing_data.get_sample_counts()
        record.freeze()
        return record
    