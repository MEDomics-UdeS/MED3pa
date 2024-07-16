"""
Contains functionality for calculating metrics based on the predicted confidence and declaration rates (MDR). 
The ``MDRCalculator`` class offers methods to assess model performance across different declaration rates,  and to extract problematic profiles under specific declaration rates.
"""
from typing import Dict, Type, Union
import numpy as np

from MED3pa.datasets import DatasetsManager
from MED3pa.detectron import DetectronExperiment, DetectronStrategy
from MED3pa.med3pa.profiles import Profile, ProfilesManager
from MED3pa.med3pa.tree import TreeRepresentation
from MED3pa.models import BaseModelManager
from MED3pa.models.classification_metrics import *

from pprint import pprint

class MDRCalculator:
    """
    Class to calculate various metrics and profiles for the MED3PA method.
    """
    @staticmethod
    def _get_min_confidence_score(dr : int, confidence_scores : np.ndarray):
        """
        Calculate the minimum confidence score based on the desired declaration rate.

        Args:
            dr (int): Desired declaratation rate as a percentage (0-100).
            confidence_scores (np.ndarray): Array of confidence scores.

        Returns:
            float: The minimum confidence level required to meet the desired declaration rate.

        Raises:
            ValueError: If dr is not in the range 0-100.
        """
        if not (0 <= dr <= 100):
                raise ValueError("Declaration rate (dr) must be between 0 and 100 inclusive.")
        
        sorted_confidence_scores = np.sort(confidence_scores)
        if dr == 0:
            min_confidence_level = 1.01
        else:
            min_confidence_level = sorted_confidence_scores[int(len(sorted_confidence_scores) * (1 - dr / 100))]
        return min_confidence_level
    
    @staticmethod
    def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, predicted_prob: np.ndarray, metrics_list : list):
        """
        Calculate a variety of metrics based on the true labels, predicted labels, and predicted probabilities.

        Args:
            y_true (np.ndarray): Array of true labels.
            y_pred (np.ndarray): Array of predicted labels.
            predicted_prob (np.ndarray): Array of predicted probabilities.
            metrics_list (list): List of metric names to be calculated.

        Returns:
            dict: A dictionary where keys are metric names and values are the calculated metric values.
        
        """
        metrics_dict = {}
        for metric_name in metrics_list:
            metric_function = ClassificationEvaluationMetrics.get_metric(metric_name)
            if metric_function:
                if metric_name in {'Auc', 'Auprc', 'Logloss'}:
                    metrics_dict[metric_name] = metric_function(y_true, predicted_prob)
                else:
                    metrics_dict[metric_name] = metric_function(y_true, y_pred)
            else:
                raise ValueError(f"Error: The metric '{metric_name}' is not supported.")
        return metrics_dict
    
    @staticmethod
    def _list_difference_by_key(list1: List[Profile], list2: List[Profile], key='node_id'):
        """
        Calculate the difference between two lists of Profile instances based on a specific key.

        Args:
            list1 (List[Profile]): First list of Profile instances.
            list2 (List[Profile]): Second list of Profile instances.
            key (str): Key to compare for differences (default is 'node_id').

        Returns:
            List[Profile]: A list containing elements from list1 that do not appear in list2 based on the specified key.
        """
        set1 = {d[key] for d in list1 if key in d}
        set2 = {d[key] for d in list2 if key in d}
        unique_to_list1 = set1 - set2
        return [d for d in list1 if d[key] in unique_to_list1]

    @staticmethod
    def _filter_by_profile(datasets_manager : DatasetsManager, path : List, set:str = 'reference', min_confidence_level:float = None):
        """
        Filters datasets based on specific profile conditions described by a path.

        Args:
            datasets_manager (DatasetsManager): The DatasetManager containing the datasets
            set (str): the set to calculate the metrics on.
            path (list): Conditions describing the profile path.
            min_confidence_level(float): possibility to filter according a minimum confidence score if specified.

        Returns:
            tuple: Filtered datasets including observations, true labels, predicted probabilities, predicted labels, and mpc values.
        """
        
        # retrieve the dataset based on the set type
        if set == 'reference':
            dataset = datasets_manager.get_dataset_by_type(dataset_type="reference", return_instance=True)
        elif set == 'testing':
            dataset = datasets_manager.get_dataset_by_type(dataset_type="testing", return_instance=True)
        else:
            raise ValueError("The set must be either the reference set or the testing set")

        # retrieve different dataset components to calculate the metrics
        x = dataset.get_observations()
        y_true = dataset.get_true_labels()
        y_pred = dataset.get_pseudo_labels()
        predicted_prob = dataset.get_pseudo_probabilities()
        confidence_scores = dataset.get_confidence_scores()
        observations = datasets_manager.get_column_labels()

        # Start with a mask that selects all rows
        mask = np.ones(len(x), dtype=bool)
        
        for condition in path:
            if condition == '*':
                continue  # Skip the root node indicator

            # Parse the condition string
            column_name, operator, value_str = condition.split(' ')
            column_index = observations.index(column_name)  # Map feature name to index
            try:
                value = float(value_str)
            except ValueError:
                # If conversion fails, the string is not a number. Handle it appropriately.
                value = value_str  # If it's supposed to be a string, leave it as string
                        
            # Apply the condition to update the mask
            if operator == '>':
                mask &= x[:, column_index] > value
            elif operator == '<':
                mask &= x[:, column_index] < value
            elif operator == '>=':
                mask &= x[:, column_index] >= value
            elif operator == '<=':
                mask &= x[:, column_index] <= value
            elif operator == '==':
                mask &= x[:, column_index] == value
            elif operator == '!=':
                mask &= x[:, column_index] != value
            else:
                raise ValueError(f"Unsupported operator '{operator}' in condition '{condition}'.")

        # Filter the data according to the path mask
        filtered_x = x[mask]
        filtered_y_true = y_true[mask]
        filtered_prob = predicted_prob[mask]
        filtered_y_pred = y_pred[mask]
        filtered_confidence_scores = confidence_scores[mask]

        # filter once again according to the min_confidence_level if specified
        if min_confidence_level is not None:
            filtered_x = filtered_x[filtered_confidence_scores>=min_confidence_level]
            filtered_y_true = filtered_y_true[filtered_confidence_scores>=min_confidence_level]
            filtered_prob = filtered_prob[filtered_confidence_scores>=min_confidence_level] if predicted_prob is not None else None
            filtered_y_pred = filtered_y_pred[filtered_confidence_scores>=min_confidence_level] if y_pred is not None else None
            filtered_confidence_scores = filtered_confidence_scores[filtered_confidence_scores>=min_confidence_level] if confidence_scores is not None else None

        return filtered_x, filtered_y_true, filtered_prob, filtered_y_pred, filtered_confidence_scores
            
    @staticmethod
    def calc_metrics_by_dr(datasets_manager: DatasetsManager, confidence_scores: np.ndarray, metrics_list: list, set = 'reference'):
        """
        Calculate metrics by declaration rates (DR), evaluating model performance at various thresholds of predicted accuracies.

        Args:
            datasets_manager (DatasetsManager): The DatasetManager containing the datasets
            metrics_list (list): List of metric names to be calculated (e.g., 'AUC', 'Accuracy').
            set (str): the set to calculate the metrics on.

        Returns:
            dict: A dictionary containing metrics computed for each declaration rate from 100% to 0%, including metrics and population percentage.
        """
        # retrieve the dataset based on the set type
        if set == 'reference':
            dataset = datasets_manager.get_dataset_by_type(dataset_type="reference", return_instance=True)
        elif set == 'testing':
            dataset = datasets_manager.get_dataset_by_type(dataset_type="testing", return_instance=True)
        else:
            raise ValueError("The set must be either the reference set or the testing set")

        # retrieve different dataset components to calculate the metrics
        y_true = dataset.get_true_labels()
        y_pred = dataset.get_pseudo_labels()
        predicted_prob = dataset.get_pseudo_probabilities()

        # initialize the dictionaries used for results storage
        metrics_by_dr = {} # global dictionary containing all the declaration rates and their corresponding metrics
        last_dr_values = {} # used to save last dr calculated metrics
        last_min_confidence_level = -1

        # for each declaration rate
        for dr in range(100, -1, -1):
            # calculate the minimum confidence level
            min_confidence_level = MDRCalculator._get_min_confidence_score(dr, confidence_scores)
            
            # if the current confidence level is different from the last one
            if last_min_confidence_level != min_confidence_level:
                
                # update the last confidence level
                last_min_confidence_level = min_confidence_level
                
                # save the confidence level in the dict of the current dr
                dr_values = {'min_confidence_level': min_confidence_level}
                
                # defines the mask to keep only data with higher min_confidence levels
                confidence_mask = confidence_scores >= min_confidence_level
                
                # save the left population percentage
                dr_values['population_percentage'] = sum(confidence_mask) / len(confidence_scores)
                dr_values['mean_confidence_level'] = np.mean(confidence_scores[confidence_mask]) if confidence_scores[confidence_mask].size>0 else None
                dr_values['Positive%'] = np.sum(y_true[confidence_mask]) / len(y_true[confidence_mask]) * 100 if \
                        len(y_true[confidence_mask]) > 0 else None
                # Calculate the metrics for the current DR
                metrics_dict = MDRCalculator._calculate_metrics(y_true[confidence_mask], y_pred[confidence_mask], predicted_prob[confidence_mask], metrics_list)
                
                # save the calculated metrics
                dr_values['metrics'] = metrics_dict
                
                # update the last dr dictionnary metrics
                last_dr_values = dr_values
                
                # save it in the global dictionnary
                metrics_by_dr[dr] = dr_values
            
            # if the min_confidence level is the same, use the last dr results 
            else:
                metrics_by_dr[dr] = last_dr_values
        
        # return the global dictionnary
        return metrics_by_dr
    
    @staticmethod
    def calc_profiles_deprecated(profiles_manager: ProfilesManager, tree: TreeRepresentation, confidence_scores: np.ndarray, min_samples_ratio: int):
        """
        Calculates profiles for different declaration rates and minimum sample ratios. This method assesses how profiles change
        across different confidence levels derived from predicted accuracies.

        Args:
            profiles_manager (ProfilesManager): Manager for storing and retrieving profile information.
            tree (TreeRepresentation): Tree structure from which profiles are derived.
            confidence_scores (np.ndarray): Array of predicted accuracy values used for thresholding profiles.
            min_samples_ratio (int): Minimum sample ratio to consider for including a profile.

        """
        # initialization of different variables
        last_profiles = tree.get_all_profiles(0, min_samples_ratio) # saves last profiles
        lost_profiles_all = [] # saves last lost profiles
        last_min_confidence_level = -1 # last min_confidence
        last_dr = 100 # last dr
        min_confidence_levels_dict = {} # saves the min_confidence_level thresholds

        # go throught all declaration rates
        for dr in range(100, -1, -1):
            # calculate the min confidence level for this dr
            min_confidence_level = MDRCalculator._get_min_confidence_score(dr, confidence_scores)
            min_confidence_levels_dict[dr] = min_confidence_level
            # if the current confidence level is different from the last one
            if last_min_confidence_level != min_confidence_level:
                # update the last min_confidence level
                last_min_confidence_level = min_confidence_level
                # get all the profiles for this min_confidence level, and min_ratio
                profiles_current = tree.get_all_profiles(min_confidence_level, min_samples_ratio)
                
                # if the last profiles are different from current profiles
                if len(last_profiles) != len(profiles_current):
                    # extract lost profiles
                    lost_profiles = MDRCalculator._list_difference_by_key(last_profiles, profiles_current)
                    lost_profiles_all.extend(lost_profiles)
                
                for insertion_dr in range (last_dr-1, dr, -1):
                    # insert these profiles in the profiles manager
                    profiles_manager.insert_profiles(insertion_dr, min_samples_ratio, profiles_current)
                    # save the lost profiles        
                    profiles_manager.insert_lost_profiles(insertion_dr, min_samples_ratio, lost_profiles_all)
                
                # update the last dr, and last profiles
                last_dr = dr
                last_profiles = profiles_current
            
            # if the current min_confidence is same as the last one
            # use the last dr results
            profiles_manager.insert_profiles(dr, min_samples_ratio, profiles_current)
            profiles_manager.insert_lost_profiles(dr, min_samples_ratio, lost_profiles_all)
        
        return min_confidence_levels_dict

    def calc_profiles(profiles_manager: ProfilesManager, tree: TreeRepresentation, datasets_manager, confidence_scores: np.ndarray, min_samples_ratio: int, set:str ='reference') -> Dict[int, float]:
        """
        Calculates profiles for different declaration rates and minimum sample ratios. This method assesses how profiles change
        across different confidence levels derived from predicted accuracies.

        Args:
            profiles_manager (ProfilesManager): Manager for storing and retrieving profile information.
            tree (TreeRepresentation): Tree structure from which profiles are derived.
            datasets_manager (DatasetsManager): Manager for handling datasets.
            confidence_scores (np.ndarray): Array of predicted accuracy values used for thresholding profiles.
            min_samples_ratio (int): Minimum sample ratio to consider for including a profile.

        Returns:
            Dict[int, float]: A dictionary with declaration rates as keys and their corresponding minimum confidence levels as values.
        """
        
        # Initialization of different variables
        all_nodes = tree.get_all_nodes()  # Retrieve all nodes from the tree
        last_profiles = all_nodes  # Initialize last profiles as all nodes
        lost_profiles_all = []  # Saves lost profiles
        last_min_confidence_level = -1  # Last min confidence level
        min_confidence_levels_dict = {}  # Saves the min_confidence_level thresholds

        # Go through all declaration rates
        for dr in range(100, -1, -1):
            
            # Calculate the min confidence level for this dr
            min_confidence_level = MDRCalculator._get_min_confidence_score(dr, confidence_scores)
            min_confidence_levels_dict[dr] = min_confidence_level

            # If the current confidence level is different from the last one
            if min_confidence_level!=last_min_confidence_level:
                
                # Update the last min confidence level
                last_min_confidence_level = min_confidence_level
                # Saves the profiles of this dr
                profiles_current = []

                # Calculate mean_ca and samples_ratio for all nodes to see if this node is eligible as a profile
                for node in all_nodes:
                    # filter the data that belongs to this node, and filter according to min_confidence_level threshold
                    _, _, _, _, filtered_confidence_scores = MDRCalculator._filter_by_profile(
                    datasets_manager, node['path'], set=set, min_confidence_level=min_confidence_level)
                    
                    # calculate the samples_ratio (pop%) and mean_confidence_level of this node, if the filtered data isnt empty
                    if len(filtered_confidence_scores) > 0:
                        samples_ratio = len(filtered_confidence_scores) / len(confidence_scores) * 100
                        mean_cconfidence = np.mean(filtered_confidence_scores) if filtered_confidence_scores.size > 0 else 0
                        # if the calculated samples_ratio and mean_confidence meet the conditions, keep this node
                        if samples_ratio >= min_samples_ratio and mean_cconfidence >= min_confidence_level:
                            profiles_current.append(node)
                    

                # If the last profiles are different from current profiles
                if len(last_profiles) != len(profiles_current):
                    # Extract lost profiles
                    lost_profiles = MDRCalculator._list_difference_by_key(last_profiles, profiles_current)
                    lost_profiles_all.extend(lost_profiles)
                    
            # Update the last profiles
            last_profiles = profiles_current

            # If the current min_confidence is same as the last one, use the last dr results
            profiles_current_ins = profiles_manager.transform_to_profiles(profiles_current)
            lost_profiles_current_ins = profiles_manager.transform_to_profiles(lost_profiles_all)
            profiles_manager.insert_profiles(dr, min_samples_ratio, profiles_current_ins)
            profiles_manager.insert_lost_profiles(dr, min_samples_ratio, lost_profiles_current_ins)

        return min_confidence_levels_dict
    
    @staticmethod
    def calc_metrics_by_profiles(profiles_manager, datasets_manager : DatasetsManager, confidence_scores: np.ndarray, min_samples_ratio: int, metrics_list, set = 'reference'):
        """
        Calculates various metrics for different profiles and declaration rates based on provided datasets.

        Args:
            profiles_manager (ProfilesManager): Manager handling profiles.
            datasets_manager (DatasetsManager): The DatasetManager containing the datasets
            set (str): the set to calculate the metrics on.
            metrics_list (list): List of metrics to calculate.

        """
        # retrieve the dataset based on the set type
        if set == 'reference':
            dataset = datasets_manager.get_dataset_by_type(dataset_type="reference", return_instance=True)
        elif set == 'testing':
            dataset = datasets_manager.get_dataset_by_type(dataset_type="reference", return_instance=True)
        else:
            raise ValueError("The set must be either the reference set or the testing set")

        # retrieve different dataset components to calculate the metrics
        all_y_true = dataset.get_true_labels()
        all_confidence_scores = confidence_scores

        dr_dict = profiles_manager.profiles_records.get(min_samples_ratio)

        # go through all profiles, for each ratio and for each dr
        if dr_dict is not None:
            # for each dr and its profiles stored in the ratio
            for dr, profiles in dr_dict.items():
                # calculate the min_confidence level
                min_confidence_level = MDRCalculator._get_min_confidence_score(dr, all_confidence_scores)
                
                # go through each profile in the profile list
                for profile in profiles:
                    x, y_true, pred_prob, y_pred, confidence_scores = MDRCalculator._filter_by_profile(datasets_manager, profile.path, set)
                    # calculate the metrics for this profile
                    confidence_mask = confidence_scores >= min_confidence_level
                    metrics_dict = MDRCalculator._calculate_metrics(y_true=y_true[confidence_mask],
                                                                    y_pred=y_pred[confidence_mask], 
                                                                    predicted_prob=pred_prob[confidence_mask],
                                                                    metrics_list=metrics_list)
                    info_dict = {}
                    # the remaining node population at the current dr compared to node population at dr = 100
                    info_dict['Node%'] = len(y_true[confidence_mask]) * 100 / len(y_true)
                    # the remaining node population at the current dr compared to the whole population at dr = 100
                    info_dict['Population%'] = len(y_true[confidence_mask]) * 100 / len(all_y_true)
                    # the mean confidence level for this profile at this dr
                    info_dict['Mean confidence level'] = np.mean(confidence_scores[confidence_mask]) * 100 if \
                        confidence_scores[confidence_mask].size > 0 else None
                    # the positive class percentage in this profile at this dr
                    info_dict['Positive%'] = np.sum(y_true[confidence_mask]) / len(y_true[confidence_mask]) * 100 if \
                        len(y_true[confidence_mask]) > 0 else None
                    # update the calculated metrics in the profile
                    profile.update_metrics_results(metrics_dict)
                    profile.update_node_information(info_dict)

    @staticmethod
    def detectron_by_profiles(datasets: DatasetsManager,
                              profiles_manager: ProfilesManager,
                              training_params: Dict,
                              base_model_manager: BaseModelManager,
                              strategies: Union[Type[DetectronStrategy], List[Type[DetectronStrategy]]],
                              samples_size: int = 20,
                              ensemble_size: int = 10,
                              num_calibration_runs: int = 100,
                              patience: int = 3,
                              allow_margin: bool = False, 
                              margin: float = 0.05, 
                              all_dr: bool = True) -> Dict:
        
        """Runs the Detectron method on the different testing set profiles.

        Args:
            datasets (DatasetsManager): The datasets manager instance.
            profiles_manager (ProfilesManager): the manager containing the profiles of the testing set.
            training_params (dict): Parameters for training the models.
            base_model_manager (BaseModelManager): The base model manager instance.
            testing_mpc_values (np.ndarray): MPC values for the testing data.
            reference_mpc_values (np.ndarray): MPC values for the reference data.
            samples_size (int, optional): Sample size for the Detectron experiment, by default 20.
            ensemble_size (int, optional): Number of models in the ensemble, by default 10.
            num_calibration_runs (int, optional): Number of calibration runs, by default 100.
            patience (int, optional): Patience for early stopping, by default 3.
            strategies (Union[Type[DetectronStrategy], List[Type[DetectronStrategy]]]): The strategies for testing disagreement.
            allow_margin (bool, optional): Whether to allow a margin in the test, by default False.
            margin (float, optional): Margin value for the test, by default 0.05.
            all_dr (bool, optional): Whether to run for all declaration rates, by default False.

        Returns:
            Dict: Dictionary of med3pa profiles with detectron results.
        """
        min_positive_ratio = min([k for k in profiles_manager.profiles_records.keys() if k >= 0])
        profiles_by_dr = profiles_manager.get_profiles(min_samples_ratio=min_positive_ratio)
        last_min_confidence_level = -1   
        confidence_scores = datasets.get_dataset_by_type(dataset_type="testing", return_instance=True).get_confidence_scores()  
        for dr, profiles in profiles_by_dr.items():
            if not all_dr and dr != 100:
                continue  # Skip all dr values except the first one if all_dr is False
            
            experiment_det = None
            min_confidence_level = MDRCalculator._get_min_confidence_score(dr, confidence_scores)
            if last_min_confidence_level != min_confidence_level:
                for profile in profiles:
                    detectron_results_dict = {}
                    
                    q_x, q_y_true, _, _, _ = MDRCalculator._filter_by_profile(datasets_manager=datasets, path=profile.path, set='testing', min_confidence_level=min_confidence_level)
                    p_x, p_y_true = datasets.get_dataset_by_type("reference")
                    if len(q_y_true) != 0:
                        if len(q_y_true) < samples_size: 
                            detectron_results_dict['Executed'] = "Not enough samples in tested profile"
                            detectron_results_dict['Tested Profile size'] = len(q_y_true)
                            detectron_results_dict['Tests Results'] = None         

                        elif 2 * samples_size > len(p_y_true):
                            detectron_results_dict['Executed'] = "Not enough samples in reference set"
                            detectron_results_dict['Tested Profile size'] = len(q_y_true)
                            detectron_results_dict['Tests Results'] = None    
                        else:
                            profile_set = DatasetsManager()
                            profile_set.set_column_labels(datasets.get_column_labels())
                            profile_set.set_from_data(dataset_type="testing", observations=q_x, true_labels=q_y_true)
                            profile_set.set_from_data(dataset_type="reference", 
                                                      observations=datasets.get_dataset_by_type(dataset_type="reference", return_instance=True).get_observations(),
                                                      true_labels=datasets.get_dataset_by_type(dataset_type="reference", return_instance=True).get_true_labels())
                            profile_set.set_from_data(dataset_type="training", 
                                                      observations=datasets.get_dataset_by_type(dataset_type="training", return_instance=True).get_observations(),
                                                      true_labels=datasets.get_dataset_by_type(dataset_type="training", return_instance=True).get_true_labels())
                            profile_set.set_from_data(dataset_type="validation", 
                                                      observations=datasets.get_dataset_by_type(dataset_type="validation", return_instance=True).get_observations(),
                                                      true_labels=datasets.get_dataset_by_type(dataset_type="validation", return_instance=True).get_true_labels())
                            
                            path_description = "*, " + " & ".join(profile.path[1:])
                            print("Running Detectron on Profile:", path_description)
                            if experiment_det is None:
                                experiment_det= DetectronExperiment.run(
                                datasets=profile_set, training_params=training_params, base_model_manager=base_model_manager,
                                samples_size=samples_size, num_calibration_runs=num_calibration_runs, ensemble_size=ensemble_size,
                                patience=patience, allow_margin=allow_margin, margin=margin)
                            else:
                                experiment_det=DetectronExperiment.run(
                                datasets=profile_set, calib_result=experiment_det.cal_record, training_params=training_params, 
                                base_model_manager=base_model_manager,
                                samples_size=samples_size, num_calibration_runs=num_calibration_runs, ensemble_size=ensemble_size,
                                patience=patience, allow_margin=allow_margin, margin=margin)
                    
                            detectron_results = experiment_det.analyze_results(strategies=strategies)
                            detectron_results_dict['Executed'] = "Yes"
                            detectron_results_dict['Tested Profile size'] = len(q_y_true)
                            detectron_results_dict['Tests Results'] = detectron_results

                    else:
                        detectron_results_dict['Executed'] = "Empty profile in test data"
                        detectron_results_dict['Tested Profile size'] = len(q_y_true)
                        detectron_results_dict['Tests Results'] = None


                    profile.update_detectron_results(detectron_results_dict)
                
                last_profiles = profiles
                last_min_confidence_level = min_confidence_level
            else:
                profiles = last_profiles

        return profiles_by_dr

