"""
this module allows for calculation of metrics by declaration rates, problematic profiles, metric by profiles..etc
"""
import numpy as np
from det3pa.models.classification_metrics import *
from det3pa.med3pa.tree import TreeRepresentation
from det3pa.med3pa.profiles import ProfilesManager

class MDRCalculator:
    """
    Class to calculate various metrics and profiles for the MED3PA method.
    """

    @staticmethod
    def calc_metrics_by_dr(y_true: np.ndarray, y_pred: np.ndarray, predicted_prob: np.ndarray, predicted_accuracies: np.ndarray, metrics_list: list):
        """
        Calculate metrics by declaration rates (DR), evaluating model performance at various thresholds of predicted accuracies.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
            predicted_prob (np.ndarray): Predicted probabilities for the positive class.
            predicted_accuracies (np.ndarray): Predicted accuracies (confidence scores) for each prediction.
            metrics_list (list): List of metric names to be calculated (e.g., 'AUC', 'Accuracy').

        Returns:
            dict: A dictionary containing metrics computed for each declaration rate from 100% to 1%, including metrics and population percentage.
        """
        metrics_by_dr = {}
        sorted_accuracies = np.sort(predicted_accuracies)
        last_dr_values = {}
        last_min_confidence_level = -1

        for dr in range(100, 0, -1):
            min_confidence_level = sorted_accuracies[int(len(sorted_accuracies) * (1 - dr / 100))]
            if last_min_confidence_level != min_confidence_level:
                last_min_confidence_level = min_confidence_level
                dr_values = {'min_confidence_level': min_confidence_level}
                dr_values['population_percentage'] = sum(predicted_accuracies >= min_confidence_level) / len(y_true)
                
                # Calculate the metrics for the current DR
                metrics_dict = {}
                for metric_name in metrics_list:
                    metric = metrics_mappings.get(metric_name, None)
                    if metric in (RocAuc, AveragePrecision):
                        metrics_dict[metric_name] = metric.calculate(
                            y_true[predicted_accuracies >= min_confidence_level],
                            predicted_prob[predicted_accuracies >= min_confidence_level])
                    else:
                        metrics_dict[metric_name] = metric.calculate(
                            y_true[predicted_accuracies >= min_confidence_level],
                            y_pred[predicted_accuracies >= min_confidence_level])
                
                dr_values['Metrics'] = metrics_dict
                last_dr_values = dr_values
                metrics_by_dr[dr] = dr_values
            else:
                metrics_by_dr[dr] = last_dr_values

        return metrics_by_dr

    @staticmethod
    def calc_profiles(profiles_manager: ProfilesManager, tree: TreeRepresentation, predicted_accuracies: np.ndarray, min_samples_ratio: int):
        """
        Calculates profiles for different declaration rates and minimum sample ratios. This method assesses how profiles change
        across different confidence levels derived from predicted accuracies.

        Args:
            profiles_manager (ProfilesManager): Manager for storing and retrieving profile information.
            tree (TreeRepresentation): Tree structure from which profiles are derived.
            predicted_accuracies (np.ndarray): Array of predicted accuracy values used for thresholding profiles.
            min_samples_ratio (int): Minimum sample ratio to consider for including a profile.

        """
        sorted_accuracies = np.sort(predicted_accuracies)
        last_profiles = tree.get_all_profiles(sorted_accuracies[0], min_samples_ratio)
        last_min_confidence_level = -1
        last_dr = 100

        for dr in range(100, -1, -1):
            min_confidence_level = 1.01 if dr == 0 else sorted_accuracies[int(len(sorted_accuracies) * (1 - dr / 100))]
            if last_min_confidence_level != min_confidence_level:
                last_min_confidence_level = min_confidence_level
                profiles_current = tree.get_all_profiles(min_confidence_level, min_samples_ratio)
                profiles_ins = ProfilesManager.transform_to_profiles(profiles_current, to_dict=False)
                profiles_manager.insert_profiles(dr, min_samples_ratio, profiles_ins)
                if last_profiles != profiles_current:
                    lost_profiles = MDRCalculator._list_difference_by_key(last_profiles, profiles_current, key='id')
                    lost_profiles_ins = ProfilesManager.transform_to_profiles(lost_profiles, to_dict=False)
                    profiles_manager.insert_lost_profiles(last_dr - 1, min_samples_ratio, lost_profiles_ins)
                else:
                    profiles_manager.insert_lost_profiles(dr, min_samples_ratio, [])
                last_dr = dr
                last_profiles = profiles_current
            else:
                profiles_manager.insert_profiles(dr, min_samples_ratio, profiles_ins)
                profiles_manager.insert_lost_profiles(dr, min_samples_ratio, [])

    @staticmethod
    def _list_difference_by_key(list1, list2, key='id'):
        """
        Calculate the difference between two lists of dictionaries based on a specific key.

        Args:
            list1 (list): First list of dictionaries.
            list2 (list): Second list of dictionaries.
            key (str): Key to compare for differences (default is 'id').

        Returns:
            list: A list containing elements from list1 that do not appear in list2 based on the specified key.
        """
        set1 = {d[key] for d in list1 if key in d}
        set2 = {d[key] for d in list2 if key in d}
        unique_to_list1 = set1 - set2
        return [d for d in list1 if d.get(key) in unique_to_list1]

    @staticmethod
    def _filter_by_profile(X, Y_true, predicted_prob, Y_pred, mpc_values, features, path):
        """
        Filters datasets based on specific profile conditions described by a path.

        Args:
            X (np.ndarray): Dataset of features.
            Y_true (np.ndarray): True labels.
            predicted_prob (np.ndarray): Predicted probabilities.
            Y_pred (np.ndarray): Predicted labels.
            mpc_values (np.ndarray): Model predicted confidence values.
            features (list): List of feature names.
            path (list): Conditions describing the profile path.

        Returns:
            tuple: Filtered datasets including features, true labels, predicted probabilities, predicted labels, and mpc values.
        """
        # Start with a mask that selects all rows
        mask = np.ones(len(X), dtype=bool)
        
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
                mask &= X[:, column_index] > value
            elif operator == '<':
                mask &= X[:, column_index] < value
            elif operator == '>=':
                mask &= X[:, column_index] >= value
            elif operator == '<=':
                mask &= X[:, column_index] <= value
            elif operator == '==':
                mask &= X[:, column_index] == value
            elif operator == '!=':
                mask &= X[:, column_index] != value
            else:
                raise ValueError(f"Unsupported operator '{operator}' in condition '{condition}'.")

        # Filter the data
        filtered_x = X[mask]
        filtered_y_true = Y_true[mask]
        filtered_prob = predicted_prob[mask]
        filtered_y_pred = Y_pred[mask]
        filtered_mpc_values = mpc_values[mask]
        return filtered_x, filtered_y_true, filtered_prob, filtered_y_pred, filtered_mpc_values
    
    @abstractmethod
    def calc_metrics_by_profiles(profiles_manager, all_x, all_y_true, all_pred_prob, all_y_pred, all_mpc_values, metrics_list):
        """
        Calculates various metrics for different profiles and declaration rates based on provided datasets.

        Args:
            profiles_manager (ProfilesManager): Manager handling profiles.
            all_x (np.ndarray): Complete dataset of features.
            all_y_true (np.ndarray): Complete dataset of true labels.
            all_pred_prob (np.ndarray): Complete dataset of predicted probabilities.
            all_y_pred (np.ndarray): Complete dataset of predicted labels.
            all_mpc_values (np.ndarray): Complete dataset of model prediction confidence values.
            metrics_list (list): List of metrics to calculate.

        """
        for min_samp_ratio, dr_dict in profiles_manager.profiles_records.items():
            for dr, profiles in dr_dict.items():
                sorted_accuracies = np.sort(all_mpc_values)
                min_confidence_level = 1.01 if dr == 0 else sorted_accuracies[int(len(sorted_accuracies) * (1 - dr / 100))]
                for profile in profiles:
                    x, y_true, pred_prob, y_pred, mpc_values = MDRCalculator._filter_by_profile(
                        all_x, all_y_true, all_pred_prob, all_y_pred, all_mpc_values, profiles_manager.features, profile.path)
                    metrics_dict = {}
                    for metric_name in metrics_list:
                        metric = metrics_mappings.get(metric_name, None)
                        if metric in (RocAuc, AveragePrecision) :
                            metrics_dict['RocAuc'] = metric.calculate(y_true[mpc_values >= min_confidence_level], pred_prob[mpc_values >= min_confidence_level])
                        else:
                            metrics_dict[metric_name] = metric.calculate(y_true[mpc_values >= min_confidence_level], y_pred[mpc_values >= min_confidence_level])

                    perc_node = len(y_true) * 100 / len(y_true)
                    perc_pop = len(y_true) * 100 / len(all_y_true)
                    metrics_dict['Node%'] = perc_node
                    metrics_dict['Population%'] = perc_pop
                    mean_ca = np.mean(mpc_values[mpc_values >= min_confidence_level]) * 100 if \
                        mpc_values[mpc_values >= min_confidence_level].size > 0 else np.NaN
                    pos_class_occurence = np.sum(y_true[mpc_values >= min_confidence_level]) / len(y_true[mpc_values >= min_confidence_level]) * 100 if \
                        len(y_true[mpc_values >= min_confidence_level]) > 0 else np.NaN
                    metrics_dict['Mean CA'] = mean_ca
                    metrics_dict['Positive%'] = pos_class_occurence
                    profile.metrics = metrics_dict

    
    