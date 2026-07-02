"""
Contains functionality for calculating metrics based on the predicted confidence and declaration rates (MDR). The
``MDRCalculator`` class offers methods to assess model performance across different declaration rates,
and to extract problematic profiles under specific declaration rates."""

import numpy as np
from copy import deepcopy
from typing import Dict

from MED3pa.datasets import MaskedDataset
from MED3pa.med3pa.profiles import ProfilesManager, Profile
from MED3pa.med3pa.tree import TreeRepresentation
from MED3pa.models.classification_metrics import *


class MDRCalculator:
    """
    Class to calculate various metrics and profiles for the MED3PA method.
    """

    @staticmethod
    def _get_min_confidence_score(dr: int, confidence_scores: np.ndarray) -> float:
        """
        Calculates the minimum confidence score based on the desired declaration rate.

        Args:
            dr (int): Desired declaration rate as a percentage (0-100).
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
            min_confidence_level = max(confidence_scores) + 1  # Higher than all confidence scores
        elif dr == 100:
            min_confidence_level = min(confidence_scores) - 1  # Lower than all confidence scores
        else:
            min_confidence_level = sorted_confidence_scores[int(len(sorted_confidence_scores) * (1 - dr / 100))]
        return min_confidence_level

    @staticmethod
    def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, predicted_prob: np.ndarray, metrics_list: list
                           ) -> dict:
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
                metrics_dict[metric_name] = metric_function(y_true=y_true, y_prob=predicted_prob, y_pred=y_pred)
            else:
                raise ValueError(f"Error: The metric '{metric_name}' is not supported.")
        return metrics_dict

    @staticmethod
    def _list_difference_by_key(list1: List[Dict], list2: List[Dict], key='node_id') -> List[Dict]:
        """
        Calculate the difference between two lists of Profile instances based on a specific key.

        Args:
            list1 (List[Dict]): First list of Profile instances.
            list2 (List[Dict]): Second list of Profile instances.
            key (str): Key to compare for differences (default is 'node_id').

        Returns:
            List[Dict]: A list containing elements from list1 that do not appear in list2 based on the specified key.
        """
        set1 = {d[key] for d in list1 if key in d}
        set2 = {d[key] for d in list2 if key in d}
        unique_to_list1 = set1 - set2
        return [d for d in list1 if d[key] in unique_to_list1]

    @staticmethod
    def _filter_by_profile(dataset: MaskedDataset, path: List, features: list,
                           confidence_scores: np.array, min_confidence_level: Optional[float ]= None):
        """
        Filters datasets based on specific profile conditions described by a path.

        Args:
            dataset (MaskedDataset): The dataset to filter.
            features (list): The list of features to filter on.
            path (list): Conditions describing the profile path.
            confidence_scores (np.array): Predicted confidence scores.
            min_confidence_level(float): Possibility to filter according a minimum confidence score if specified.

        Returns:
            tuple: Filtered datasets including observations, true labels, predicted probabilities, predicted labels, and mpc values.
        """

        # retrieve different dataset components to calculate the metrics
        x = dataset.get_observations()
        y_true = dataset.get_true_labels()
        y_pred = dataset.get_pseudo_labels()
        predicted_prob = dataset.get_pseudo_probabilities()

        # Start with a mask that selects all rows
        mask = np.ones(len(x), dtype=bool)

        for condition in path:
            if condition == '*':
                continue  # Skip the root node indicator

            # Parse the condition string
            column_name, operator, value_str = condition.split(' ')
            column_index = features.index(column_name)  # Map feature name to index
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
        if predicted_prob is not None:
            filtered_prob = predicted_prob[mask]
        else:
            filtered_prob = None

        if y_pred is not None:
            filtered_y_pred = y_pred[mask]
        else:
            filtered_y_pred = None

        filtered_confidence_scores = confidence_scores[mask]

        # filter once again according to the min_confidence_level if specified
        if min_confidence_level is not None:
            filtered_x = filtered_x[filtered_confidence_scores >= min_confidence_level]
            filtered_y_true = filtered_y_true[filtered_confidence_scores >= min_confidence_level]
            filtered_prob = filtered_prob[
                filtered_confidence_scores >= min_confidence_level] if predicted_prob is not None else None
            filtered_y_pred = filtered_y_pred[
                filtered_confidence_scores >= min_confidence_level] if y_pred is not None else None
            filtered_confidence_scores = filtered_confidence_scores[
                filtered_confidence_scores >= min_confidence_level]

        return filtered_x, filtered_y_true, filtered_prob, filtered_y_pred, filtered_confidence_scores

    @staticmethod
    def calc_metrics_by_dr(dataset: MaskedDataset, confidence_scores: np.ndarray, metrics_list: list
                           ) -> Dict[int, Dict]:
        """
        Calculate metrics by declaration rates (DR), evaluating model performance at various thresholds of predicted
        accuracies.

        Args:
            dataset (MaskedDataset): The dataset to filter.
            confidence_scores (np.ndarray): the confidence scores used for filtering.
            metrics_list (list): List of metric names to be calculated (e.g., 'AUC', 'Accuracy').

        Returns:
            Dict: A dictionary containing metrics computed for each declaration rate from 100% to 0%, including metrics
                and population percentage.
        """

        # retrieve different dataset components to calculate the metrics
        y_true = dataset.get_true_labels()
        y_pred = dataset.get_pseudo_labels()
        predicted_prob = dataset.get_pseudo_probabilities()

        # initialize the dictionaries used for results storage
        metrics_by_dr = {}  # global dictionary containing all the declaration rates and their corresponding metrics
        last_dr_values = {}  # used to save last dr calculated metrics
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
                dr_values['mean_confidence_level'] = np.mean(confidence_scores[confidence_mask]) if confidence_scores[
                                                                                                        confidence_mask].size > 0 else None
                dr_values['Positive%'] = np.sum(y_true[confidence_mask]) / len(y_true[confidence_mask]) * 100 if \
                    len(y_true[confidence_mask]) > 0 else None
                # Calculate the metrics for the current DR
                metrics_dict = MDRCalculator._calculate_metrics(y_true[confidence_mask], y_pred[confidence_mask],
                                                                predicted_prob[confidence_mask], metrics_list)

                # save the calculated metrics
                dr_values['metrics'] = metrics_dict

                # update the last dr dictionary metrics
                last_dr_values = dr_values

                # save it in the global dictionary
                metrics_by_dr[dr] = dr_values

            # if the min_confidence level is the same, use the last dr results 
            else:
                metrics_by_dr[dr] = last_dr_values

        # return the global dictionary
        return metrics_by_dr

    @staticmethod
    def calc_profiles(profiles_manager: ProfilesManager, tree: TreeRepresentation, dataset: MaskedDataset,
                      features: list, confidence_scores: np.ndarray, min_samples_ratio: int,
                      metrics_list: List) -> Dict[int, float]:
        """
        Calculates profiles for different declaration rates and minimum sample ratios. This method assesses how profiles
        change across different confidence levels derived from predicted accuracies.

        Args:
            profiles_manager (ProfilesManager): Manager for storing and retrieving profile information.
            tree (TreeRepresentation): Tree structure from which profiles are derived.
            dataset (MaskedDataset): The dataset to filter.
            features (list): the list of features to filter on.
            confidence_scores (np.ndarray): Array of predicted confidence values used for thresholding profiles.
            min_samples_ratio (int): Minimum sample ratio to consider for including a profile.
            metrics_list (List): List of metrics to calculate.

        Returns:
            Dict[int, float]: A dictionary with declaration rates as keys and their corresponding minimum confidence levels as values.
        """

        # Initialization of different variables
        all_nodes = tree.get_all_nodes()  # Retrieve all nodes from the tree
        last_profiles = all_nodes  # Initialize last profiles as all nodes
        lost_profiles_all = []  # Saves lost profiles
        last_min_confidence_level = None  # Last min confidence level
        min_confidence_levels_dict = {}  # Saves the min_confidence_level thresholds

        # Go through all declaration rates
        for dr in range(100, -1, -1):

            # Calculate the min confidence level for this dr
            min_confidence_level = MDRCalculator._get_min_confidence_score(dr, confidence_scores)
            min_confidence_levels_dict[dr] = min_confidence_level

            # If the current confidence level is different from the last one
            if min_confidence_level != last_min_confidence_level:

                # Update the last min confidence level
                last_min_confidence_level = min_confidence_level
                # Saves the profiles of this dr
                profiles_current = []

                # Calculate mean confidence and samples_ratio for all nodes to see if this node is eligible as a profile
                for node in all_nodes:
                    # filter the data that belongs to this node, and filter according to min_confidence_level threshold
                    _, _, _, _, filtered_confidence_scores = MDRCalculator._filter_by_profile(
                        dataset, node.path, features=features, confidence_scores=confidence_scores, min_confidence_level=min_confidence_level)

                    # calculate the samples_ratio (pop%) and mean_confidence_level of this node
                    if len(filtered_confidence_scores) > 0:
                        samples_ratio = len(filtered_confidence_scores) / len(confidence_scores) * 100
                        max_confidence = float(np.max(
                            filtered_confidence_scores)) if filtered_confidence_scores.size > 0 else 0
                        # if the calculated samples_ratio and max_confidence meet the conditions, keep this node
                        if samples_ratio >= min_samples_ratio and max_confidence >= min_confidence_level:
                            # Calculate metrics for this profile
                            MDRCalculator.__calc_metrics_in_profile(profile=node,
                                                                    dataset=dataset,
                                                                    min_confidence_level=min_confidence_level,
                                                                    features=features,
                                                                    confidence_scores=confidence_scores,
                                                                    metrics_list=metrics_list)
                            profiles_current.append(deepcopy(node))

                # If the last profiles are different from current profiles
                if len(last_profiles) != len(profiles_current):
                    # Extract lost profiles
                    lost_profiles = MDRCalculator._list_difference_by_key(last_profiles, profiles_current)
                    lost_profiles_all.extend(lost_profiles)

                # Update the last profiles
                last_profiles = profiles_current

            # If the current min_confidence is same as the last one, use the last dr results
            # profiles_current_ins = profiles_manager.transform_to_profiles(profiles_current)
            # lost_profiles_current_ins = profiles_manager.transform_to_profiles(lost_profiles_all)
            profiles_manager.insert_profiles(dr, min_samples_ratio, profiles_current)
            profiles_manager.insert_lost_profiles(dr, min_samples_ratio, lost_profiles_all)

        return min_confidence_levels_dict

    @staticmethod
    def __calc_metrics_in_profile(profile: Profile, dataset: MaskedDataset, min_confidence_level: float,
                                  features: List, confidence_scores: np.ndarray, metrics_list: List) -> None:
        """
        Calculates various metrics for different profiles and declaration rates based on provided datasets.

        Args:
            profile (ProfilesManager): Manager handling profiles.
            dataset (MaskedDataset): the dataset to use.
            features (List): The list of features to filter on.
            confidence_scores (np.ndarray): Array of predicted accuracy values used for thresholding profiles.
            metrics_list (List): List of metrics to calculate.

        """
        # retrieve different dataset components to calculate the metrics
        total_number = len(dataset.get_true_labels())

        x, y_true, pred_prob, y_pred, confidence_scores = MDRCalculator._filter_by_profile(
                                                                        dataset=dataset,
                                                                        path=profile.path,
                                                                        features=features,
                                                                        confidence_scores=confidence_scores)
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
        info_dict['Population%'] = len(y_true[confidence_mask]) * 100 / total_number
        # the mean confidence level for this profile at this dr
        info_dict['Mean confidence level'] = float(np.mean(confidence_scores[confidence_mask])) * 100 if \
            confidence_scores[confidence_mask].size > 0 else None
        # the positive class percentage in this profile at this dr
        info_dict['Positive%'] = np.sum(y_true[confidence_mask]) / len(y_true[confidence_mask]) * 100 if \
            len(y_true[confidence_mask]) > 0 else None
        # update the calculated metrics in the profile
        profile.update_metrics_results(metrics_dict)
        profile.update_node_information(info_dict)

    # @staticmethod
    # def calc_metrics_by_profiles(profiles_manager, dataset: MaskedDataset, features: List,
    #                              confidence_scores: np.ndarray, min_samples_ratio: int, metrics_list: List) -> None:
    #     """
    #     Calculates various metrics for different profiles and declaration rates based on provided datasets.
    #
    #     Args:
    #         profiles_manager (ProfilesManager): Manager handling profiles.
    #         dataset (MaskedDataset): the dataset to use.
    #         features (List): The list of features to filter on.
    #         confidence_scores (np.ndarray): Array of predicted accuracy values used for thresholding profiles.
    #         min_samples_ratio (int): Minimum sample ratio to consider for including a profile.
    #         metrics_list (List): List of metrics to calculate.
    #
    #     """
    #     # retrieve different dataset components to calculate the metrics
    #     all_y_true = dataset.get_true_labels()
    #     all_confidence_scores = confidence_scores
    #
    #     dr_dict = profiles_manager.profiles_records.get(min_samples_ratio)
    #
    #     # go through all profiles, for each ratio and for each dr
    #     if dr_dict is not None:
    #         # for each dr and its profiles stored in the ratio
    #         for dr, profiles in dr_dict.items():
    #             # calculate the min_confidence level
    #             min_confidence_level = MDRCalculator._get_min_confidence_score(dr, all_confidence_scores)
    #
    #             # go through each profile in the profile list
    #             for profile in profiles:
    #                 x, y_true, pred_prob, y_pred, confidence_scores = MDRCalculator._filter_by_profile(dataset,
    #                                                                                                    profile.path,
    #                                                                                                    features)
    #                 # calculate the metrics for this profile
    #                 confidence_mask = confidence_scores >= min_confidence_level
    #                 metrics_dict = MDRCalculator._calculate_metrics(y_true=y_true[confidence_mask],
    #                                                                 y_pred=y_pred[confidence_mask],
    #                                                                 predicted_prob=pred_prob[confidence_mask],
    #                                                                 metrics_list=metrics_list)
    #                 info_dict = {}
    #                 # the remaining node population at the current dr compared to node population at dr = 100
    #                 info_dict['Node%'] = len(y_true[confidence_mask]) * 100 / len(y_true)
    #                 # the remaining node population at the current dr compared to the whole population at dr = 100
    #                 info_dict['Population%'] = len(y_true[confidence_mask]) * 100 / len(all_y_true)
    #                 # the mean confidence level for this profile at this dr
    #                 info_dict['Mean confidence level'] = np.mean(confidence_scores[confidence_mask]) * 100 if \
    #                     confidence_scores[confidence_mask].size > 0 else None
    #                 # the positive class percentage in this profile at this dr
    #                 info_dict['Positive%'] = np.sum(y_true[confidence_mask]) / len(y_true[confidence_mask]) * 100 if \
    #                     len(y_true[confidence_mask]) > 0 else None
    #                 # update the calculated metrics in the profile
    #                 profile.update_metrics_results(metrics_dict)
    #                 profile.update_node_information(info_dict)
