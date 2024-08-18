"""
Compares between two experiments, either two ``Med3paExperiment`` or two ``Med3paDetectronExperiment`` 
"""
import json
import os
from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np
from sklearn.model_selection import train_test_split

from MED3pa.med3pa.models import *
from MED3pa.med3pa.uncertainty import *
from MED3pa.models.base import BaseModelManager
from MED3pa.med3pa.experiment import Med3paResults

class Med3paComparison:
    """
    Class to compare the results of two Med3paExperiment instances.
    """
    def __init__(self, results1_path: str, results2_path: str) -> None:
        self.results1_path = os.path.abspath(results1_path)
        self.results2_path = os.path.abspath(results2_path)
        self.profiles_metrics_comparaison = {}
        self.profiles_detectron_comparaison = {}
        self.global_metrics_comparaison = {}
        self.models_evaluation_comparaison = {}
        self.shared_profiles = {}  # New variable to store shared profiles
        self.config_file = {}
        self.compare_profiles = True
        self.compare_detectron = False
        self.mode = ""
        self._check_experiment_name()

    def identify_shared_profiles(self):
        """
        Identifies the shared profiles between the two experiments and stores them in shared_profiles.
        """
        profiles_file_1 = os.path.join(self.results1_path, 'test', 'profiles.json')
        profiles_file_2 = os.path.join(self.results2_path, 'test', 'profiles.json')

        with open(profiles_file_1, 'r') as f1, open(profiles_file_2, 'r') as f2:
            profiles1 = json.load(f1)
            profiles2 = json.load(f2)

        shared = {}

        for samples_ratio, dr_dict in profiles1.items():
            if samples_ratio in profiles2:  # Only proceed if samples_ratio is in both profiles
                if samples_ratio not in shared:
                    shared[samples_ratio] = {}
                for dr, profiles in dr_dict.items():
                    if dr in profiles2[samples_ratio]:  # Only proceed if dr is in both profiles
                        for profile in profiles:
                            profile_path = " / ".join(profile["path"])
                            # Check if the profile_path exists in both profiles1 and profiles2
                            matching_profile = next((p for p in profiles2[samples_ratio][dr] if p["path"] == profile["path"]), None)
                            if matching_profile:
                                if profile_path not in shared[samples_ratio]:
                                    shared[samples_ratio][profile_path] = profile["path"]

        self.shared_profiles = shared  # Store shared profiles

    def _check_experiment_name(self) -> None:
        """
        Checks if the experiment_name in the config_file of both results paths is the same.
        If not, raises a ValueError. Also sets the flag for Detectron comparison if applicable.
        """
        config_file_1 = os.path.join(self.results1_path, 'experiment_config.json')
        config_file_2 = os.path.join(self.results2_path, 'experiment_config.json')

        with open(config_file_1, 'r') as f1, open(config_file_2, 'r') as f2:
            config1 = json.load(f1)
            config2 = json.load(f2)

        if config1['experiment_name'] not in ["Med3paDetectronExperiment", "Med3paExperiment"]:
            raise ValueError("Only Med3paDetectronExperiments & Med3paExperiments can be compared")
        
        if config1['experiment_name'] != config2['experiment_name']:
            raise ValueError("The two results are not from the same experiment.")
        
        if config1['experiment_name'] == 'Med3paDetectronExperiment':
            self.compare_detectron = True
    
    def is_comparable(self) -> bool:
        """
        Determines whether the two experiments can be compared based on the given criteria.
        Two experiments can be compared if either:
        - The datasets are different, but the base model and experiment parameters are the same.
        - The base model is different, but the datasets and experiment parameters are the same.

        Returns:
            bool: True if the experiments can be compared, False otherwise.
        """
        # Load the config files if they haven't been compared yet
        if not self.config_file:
            self.compare_config()

        datasets_different = self.config_file['datasets']['different']
        datasets_different_sets = self.config_file['datasets']['different_datasets']
        base_model_different = self.config_file['base_model']['different']
        
        if self.compare_detectron:
            # Extract med3pa_detectron_params for comparison, excluding apc_model and ipc_model
            params1 = self.config_file['med3pa_detectron_params']['med3pa_detectron_params1'].copy()
            params2 = self.config_file['med3pa_detectron_params']['med3pa_detectron_params2'].copy()
            # Remove apc_model and ipc_model from comparison
            params1['med3pa_params'].pop('apc_model', None)
            params1['med3pa_params'].pop('ipc_model', None)
            params2['med3pa_params'].pop('apc_model', None)
            params2['med3pa_params'].pop('ipc_model', None)
        else:
            # Extract med3pa_params for comparison, excluding apc_model and ipc_model
            params1 = self.config_file['med3pa_params']['med3pa_params1'].copy()
            params2 = self.config_file['med3pa_params']['med3pa_params2'].copy()
            params1.pop('apc_model', None)
            params1.pop('ipc_model', None)
            params2.pop('apc_model', None)
            params2.pop('ipc_model', None)


        params_different = (params1 != params2)
        # Check the conditions for comparability
        can_compare = False
        # First condition: params are the same, base model is the same, only the testing_set is different
        if not params_different and not base_model_different and datasets_different_sets == ['testing_set']:
            can_compare = True
        # Second condition: base model is different, params are the same, datasets are the same or only differ in training and validation sets
        elif base_model_different and not params_different and (not datasets_different or set(datasets_different_sets) <= {'training_set', 'validation_set', 'reference_set'}):
            can_compare = True
            
        if can_compare:
            if self.compare_detectron:
                self.mode = self.config_file['med3pa_detectron_params']['med3pa_detectron_params1']['med3pa_params']['mode']
            else:
                self.mode = self.config_file['med3pa_params']['med3pa_params1']['mode']

        return can_compare
    
    def _check_experiment_tree(self) -> None:
        """
        Checks if the experiment trees in the results paths are the same.
        If they are, sets the flag for profile comparison.
        """
        tree_file_1 = os.path.join(self.results1_path, 'test', 'tree.json')
        tree_file_2 = os.path.join(self.results2_path, 'test', 'tree.json')

        with open(tree_file_1, 'r') as f1, open(tree_file_2, 'r') as f2:
            tree1 = json.load(f1)
            tree2 = json.load(f2)

        if tree1 == tree2:
            self.compare_profiles = True

    def compare_profiles_metrics(self):
        """
        Compares profile metrics between two sets of results and stores them in a dictionary,
        using only the shared profiles.
        """
        combined = {}
        profiles_file_1 = os.path.join(self.results1_path, 'test', 'profiles.json')
        profiles_file_2 = os.path.join(self.results2_path, 'test', 'profiles.json')

        with open(profiles_file_1, 'r') as f1, open(profiles_file_2, 'r') as f2:
            profiles1 = json.load(f1)
            profiles2 = json.load(f2)

        for samples_ratio, profiles_dict in self.shared_profiles.items():
            combined[samples_ratio] = {}
            for profile_path_list in profiles_dict.values():
                profile_path = " / ".join(profile_path_list)  # Convert the list to a string

                # Extract possible drs (decision rules) where profiles match in profiles1
                drs = [dr for dr, profiles in profiles1[samples_ratio].items()]
                for dr in drs:
                    # Attempt to find matching profiles in both profiles1 and profiles2
                    matching_profile_1 = next((p for p in profiles1[samples_ratio][dr] if " / ".join(p["path"]) == profile_path), None)
                    matching_profile_2 = next((p for p in profiles2[samples_ratio][dr] if " / ".join(p["path"]) == profile_path), None)

                    if profile_path not in combined[samples_ratio]:
                        combined[samples_ratio][profile_path] = {}

                    combined[samples_ratio][profile_path][dr] = {
                        'metrics_1': matching_profile_1["metrics"] if matching_profile_1 else None,
                        'metrics_2': matching_profile_2["metrics"] if matching_profile_2 else None
                    }

        self.profiles_metrics_comparaison = combined


    def compare_profiles_detectron_results(self):
        """
        Compares Detectron results between two sets of profiles and stores them in a dictionary,
        using only the shared profiles.
        """
        combined = {}

        profiles_file_1 = os.path.join(self.results1_path, 'test', 'profiles.json')
        profiles_file_2 = os.path.join(self.results2_path, 'test', 'profiles.json')

        with open(profiles_file_1, 'r') as f1, open(profiles_file_2, 'r') as f2:
            profiles1 = json.load(f1)
            profiles2 = json.load(f2)

        for samples_ratio, profiles_dict in self.shared_profiles.items():
            combined = {}
            for profile_path_list in profiles_dict.values():
                profile_path = " / ".join(profile_path_list)  # Convert the list to a string

                # Attempt to find matching profiles in both profiles1 and profiles2
                matching_profile_1 = next((p for p in profiles1[samples_ratio]["100"] if " / ".join(p["path"]) == profile_path), None)
                matching_profile_2 = next((p for p in profiles2[samples_ratio]["100"] if " / ".join(p["path"]) == profile_path), None)

                if profile_path not in combined:
                    combined[profile_path] = {}

                combined[profile_path]['detectron_results_1'] = matching_profile_1["detectron_results"] if matching_profile_1 else None
                combined[profile_path]['detectron_results_2'] = matching_profile_2["detectron_results"] if matching_profile_2 else None

        self.profiles_detectron_comparaison = combined


    def compare_global_metrics(self):
        """
        Compares global metrics between two sets of results and stores them in a dictionary.
        """
        combined = {}
        file_1 = os.path.join(self.results1_path, 'test', 'metrics_dr.json')
        file_2 = os.path.join(self.results2_path, 'test', 'metrics_dr.json')

        with open(file_1, 'r') as f1, open(file_2, 'r') as f2:
            dr1 = json.load(f1)
            dr2 = json.load(f2)
        
        for dr in range(100, -1, -1):  # Iterating from 100 to 0
            dr_str = str(dr)
            combined[dr_str] = {}
            
            if dr_str in dr1:
                combined[dr_str]['metrics_dr_1'] = dr1[dr_str]
            if dr_str in dr2:
                combined[dr_str]['metrics_dr_2'] = dr2[dr_str]

        self.global_metrics_comparaison = combined
    
    def compare_models_evaluation(self):
        """
        Compares IPC and APC evaluation between two experiments.
        """
        combined = {}
        file_1 = os.path.join(self.results1_path, 'test', 'models_evaluation.json')
        file_2 = os.path.join(self.results2_path, 'test', 'models_evaluation.json')

        with open(file_1, 'r') as f1, open(file_2, 'r') as f2:
            models1 = json.load(f1)
            models2 = json.load(f2)
        
        if "IPC_evaluation" in models1 and "IPC_evaluation" in models2:
            combined['IPC_evaluation1'] = models1["IPC_evaluation"]
            combined['IPC_evaluation2'] = models2["IPC_evaluation"]

        if "APC_evaluation" in models1 and "APC_evaluation" in models2:
            combined['APC_evaluation1'] = models1["APC_evaluation"]
            combined['APC_evaluation2'] = models2["APC_evaluation"]


        self.models_evaluation_comparaison = combined

    def compare_config(self):
        """
        Compares the config files of the two experiments.
        """
        combined = {}
        config_file_1 = os.path.join(self.results1_path, 'experiment_config.json')
        config_file_2 = os.path.join(self.results2_path, 'experiment_config.json')

        with open(config_file_1, 'r') as f1, open(config_file_2, 'r') as f2:
            config1 = json.load(f1)
            config2 = json.load(f2)

        combined['datasets'] = {}
        dataset_keys = ['training_set', 'validation_set', 'reference_set', 'testing_set']
        different_datasets = []

        if config1["datasets"] == config2["datasets"]:
            combined['datasets']['different'] = False
        else:
            combined['datasets']['different'] = True

        for key in dataset_keys:
            if config1["datasets"].get(key) != config2["datasets"].get(key):
                different_datasets.append(key)

        combined['datasets']['different_datasets'] = different_datasets

        combined['datasets']['datasets1'] = config1["datasets"]
        combined['datasets']['datasets2'] = config2["datasets"]

        combined['base_model'] = {}

        if config1["base_model"] == config2["base_model"]:
            combined['base_model']['different'] = False
        else:
            combined['base_model']['different'] = True

        combined['base_model']['base_model1'] = config1["base_model"]
        combined['base_model']['base_model2'] = config2["base_model"]

        if not self.compare_detectron :
            
            combined['med3pa_params'] = {}

            if config1["med3pa_params"] == config2["med3pa_params"]:
                combined['med3pa_params']['different'] = False
            else:
                combined['med3pa_params']['different'] = True

            combined['med3pa_params']['med3pa_params1'] = config1["med3pa_params"]
            combined['med3pa_params']['med3pa_params2'] = config2["med3pa_params"]
        
        else:

            combined['med3pa_detectron_params'] = {}

            if config1["med3pa_detectron_params"] == config2["med3pa_detectron_params"]:
                combined['med3pa_detectron_params']['different'] = False
            else:
                combined['med3pa_detectron_params']['different'] = True

            combined['med3pa_detectron_params']['med3pa_detectron_params1'] = config1["med3pa_detectron_params"]
            combined['med3pa_detectron_params']['med3pa_detectron_params2'] = config2["med3pa_detectron_params"]


        self.config_file = combined

    def compare_experiments(self):
        """
        Compares the experiments by global metrics, profiles, and Detectron results if applicable.
        """
        if not self.is_comparable():
            raise ValueError("The two experiments cannot be compared based on the provided criteria.")
        
        self.compare_global_metrics()
        if self.mode in ['apc', 'mpc']:
            self.identify_shared_profiles() 
            if self.compare_profiles:
                self.compare_profiles_metrics()
            if self.compare_detectron:
                self.compare_profiles_detectron_results()
        
        self.compare_config()
        self.compare_models_evaluation()

    def save(self, directory_path: str) -> None:
        """
        Saves the comparison results to a specified directory.

        Args:
            directory_path (str): The directory where the comparison results will be saved.
        """
        
        if not self.is_comparable():
            raise ValueError("The two experiments cannot be compared based on the provided criteria.")
        
        # Ensure the main directory exists
        os.makedirs(directory_path, exist_ok=True)

        global_comparaison_path = os.path.join(directory_path, 'global_metrics_comparaison.json')
        with open(global_comparaison_path, 'w') as f:
                json.dump(self.global_metrics_comparaison, f, indent=4)

        config_path = os.path.join(directory_path, 'experiment_config_comparaison.json')
        with open(config_path, 'w') as f:
                json.dump(self.config_file, f, indent=4)
        
        evaluation_path = os.path.join(directory_path, 'models_evaluation_comparaison.json')
        with open(evaluation_path, 'w') as f:
                json.dump(self.models_evaluation_comparaison, f, indent=4)

        if self.profiles_detectron_comparaison != {} and self.compare_detectron:
            profiles_detectron_path = os.path.join(directory_path, 'profiles_detectron_comparaison.json')
            with open(profiles_detectron_path, 'w') as f:
                json.dump(self.profiles_detectron_comparaison, f, indent=4)
        
        if self.profiles_metrics_comparaison != {} and self.compare_profiles:
            profiles_metrics_path = os.path.join(directory_path, 'profiles_metrics_comparaison.json')
            with open(profiles_metrics_path, 'w') as f:
                json.dump(self.profiles_metrics_comparaison, f, indent=4)