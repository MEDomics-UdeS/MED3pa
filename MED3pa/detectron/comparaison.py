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
        self.model_evaluation_comparaison = {}
        self.rejection_counts_comparaison = {}
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
        Compares detectron results two sets of results and stores them in a dictionary.
        """
        combined = {}
        file_1 = os.path.join(self.results1_path, 'detectron_results.json')
        file_2 = os.path.join(self.results2_path, 'detectron_results.json')

        with open(file_1, 'r') as f1, open(file_2, 'r') as f2:
            detectron1 = json.load(f1)
            detectron2 = json.load(f2)

        combined['detectron_results1'] = detectron1
        combined['detectron_results2'] = detectron2

        comparison_results = {}
        for r1 in detectron1:
            for r2 in detectron2:
                if r1["Strategy"] == r2["Strategy"]:
                    strategy = r1["Strategy"]
                    strategy_dict = {}
                    if "p_value" in r1 and "p_value" in r2:
                        strategy_dict["detectron_results1"] = r1["p_value"]
                        strategy_dict["detectron_results2"] = r2["p_value"]
                        strategy_dict["comparison_criteria"] = "p_value"
                        strategy_dict["best"] = "detectron_results2" if r2["p_value"] > r1["p_value"] else "detectron_results1"

                    if "shift_probability" in r1 and "shift_probability" in r2:
                        strategy_dict["detectron_results1"] = r1["shift_probability"]
                        strategy_dict["detectron_results2"] = r2["shift_probability"]
                        strategy_dict["comparison_criteria"] = "shift_probability"
                        strategy_dict["best"] = "detectron_results2" if r2["shift_probability"] < r1["shift_probability"] else "detectron_results1"
                    
                    if strategy not in comparison_results:
                        comparison_results[strategy] = {}
                        comparison_results[strategy] = strategy_dict
                    
        self.detectron_results_comparaison = comparison_results

    def compare_evaluation(self):
        """
        Compares model evaluations two sets of results and stores them in a dictionary.
        """
        combined = {}
        file_1 = os.path.join(self.results1_path, 'model_evaluation.json')
        file_2 = os.path.join(self.results2_path, 'model_evaluation.json')

        with open(file_1, 'r') as f1, open(file_2, 'r') as f2:
            detectron1 = json.load(f1)
            detectron2 = json.load(f2)

        combined['model_evaluation1'] = detectron1
        combined['model_evaluation2'] = detectron2
        comparison = {}
        metrics = detectron1["test"].keys()  # Assuming both experiments have the same metrics
        
        for metric in metrics:
            comparison[metric] = {
                "detectron_results1": detectron1["test"][metric],
                "detectron_results2": detectron2["test"][metric],
                "best": "detectron_results1" if detectron1["test"][metric] > detectron2["test"][metric] else "detectron_results2"
            }
            
        self.model_evaluation_comparaison = comparison

    def compare_counts(self):
        """
        Compares rejection two sets of results and stores them in a dictionary.
        """
        combined = {}
        file_1 = os.path.join(self.results1_path, 'rejection_counts.json')
        file_2 = os.path.join(self.results2_path, 'rejection_counts.json')

        with open(file_1, 'r') as f1, open(file_2, 'r') as f2:
            detectron1 = json.load(f1)
            detectron2 = json.load(f2)

        combined['rejection_counts1'] = detectron1
        combined['rejection_counts2'] = detectron2

        self.rejection_counts_comparaison = combined

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
        
        if config1["datasets"] == config2["datasets"]:
            combined['datasets']['different'] = False
        else:
            combined['datasets']['different'] = True

        combined['datasets']['datasets1'] = config1["datasets"]
        combined['datasets']['datasets2'] = config2["datasets"]

        combined['base_model'] = {}

        if config1["base_model"] == config2["base_model"]:
            combined['base_model']['different'] = False
        else:
            combined['base_model']['different'] = True

        combined['base_model']['base_model1'] = config1["base_model"]
        combined['base_model']['base_model2'] = config2["base_model"]

        combined['detectron_params'] = {}

        if config1["detectron_params"] == config2["detectron_params"]:
            combined['detectron_params']['different'] = False
        else:
            combined['detectron_params']['different'] = True

        combined['detectron_params']['detectron_params1'] = config1["detectron_params"]
        combined['detectron_params']['detectron_params2'] = config2["detectron_params"]

        self.config_file = combined
        
    def compare_experiments(self):
        """
        Compares the experiments by detectron_results.
        """
        self.compare_detectron_results()
        self.compare_counts()
        self.compare_evaluation()
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

        eval_path = os.path.join(directory_path, 'model_evaluation_comparaison.json')
        with open(eval_path, 'w') as f:
                json.dump(self.model_evaluation_comparaison, f, indent=4)

        counts_path = os.path.join(directory_path, 'rejection_counts_comparaison.json')
        with open(counts_path, 'w') as f:
                json.dump(self.rejection_counts_comparaison, f, indent=4)
