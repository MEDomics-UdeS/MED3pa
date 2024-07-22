"""
Compares between two ``DetectronExperiment``.
"""
import json
import os
from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np
from sklearn.model_selection import train_test_split

class DetectronComparison:
    """
    Class to compare the results of two Med3paExperiment instances.
    """
    def __init__(self, results1_path: str, results2_path: str) -> None:
        self.results1_path = os.path.abspath(results1_path)
        self.results2_path = os.path.abspath(results2_path)
        self.detectron_results_comparaison = {}
        self.config_file = {}
        self._check_experiment_name()

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

        if config1['experiment_name'] != "DetectronExperiment":
            raise ValueError("Only DetectronExperiment can be compared using this class")
        
        if config1['experiment_name'] != config2['experiment_name']:
            raise ValueError("The two results are not from the same experiment.")
        
    def compare_detectron_results(self):
        """
        Compares profile metrics between two sets of results and stores them in a dictionary.
        """
        combined = {}
        file_1 = os.path.join(self.results1_path, 'detectron_results.json')
        file_2 = os.path.join(self.results2_path, 'detectron_results.json')

        with open(file_1, 'r') as f1, open(file_2, 'r') as f2:
            detectron1 = json.load(f1)
            detectron2 = json.load(f2)

        combined['detectron_results1'] = detectron1
        combined['detectron_results2'] = detectron2

        self.detectron_results_comparaison = combined

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

        combined['datasets1'] = config1["datasets"]
        combined['datasets2'] = config2["datasets"]

        combined['base_model1'] = config1["base_model"]
        combined['base_model2'] = config2["base_model"]

        combined['experiment_params1'] = config1["experiment_params"]
        combined['experiment_params2'] = config2["experiment_params"]

        self.config_file = combined

    def compare_experiments(self):
        """
        Compares the experiments by detectron_results.
        """
        self.compare_detectron_results()
        self.compare_config()

    def save(self, directory_path: str) -> None:
        """
        Saves the comparison results to a specified directory.

        Args:
            directory_path (str): The directory where the comparison results will be saved.
        """
        # Ensure the main directory exists
        os.makedirs(directory_path, exist_ok=True)

        global_comparaison_path = os.path.join(directory_path, 'detectron_results_comparaison.json')
        with open(global_comparaison_path, 'w') as f:
                json.dump(self.detectron_results_comparaison, f, indent=4)
        
        config_path = os.path.join(directory_path, 'experiment_config_comparaison.json')
        with open(config_path, 'w') as f:
                json.dump(self.config_file, f, indent=4)
