"""
This module stores and manages the results of the MED3pa experiments.
It includes the ``Med3paRecord`` class, responsible for storing and managing results for each set,
and the ``Med3paResult`` class, responsible for storing and managing all results of the experiment.
"""

import datetime
import json
import numpy as np
import os
from typing import Any, Dict, Optional, TextIO

from MED3pa.datasets import MaskedDataset
from MED3pa.med3pa.models import APCModel, IPCModel, MPCModel, MpcStrategy
from MED3pa.med3pa.profiles import Profile, ProfilesManager
from MED3pa.med3pa.tree import TreeRepresentation


def to_serializable(obj: Any, additional_arg: Any = None) -> Any:
    """
    Convert an object to a JSON-serializable format.
    Args:
        obj (Any): The object to convert.
        additional_arg (Any): Additional arguments to serialize
    Returns:
        Any: The JSON-serializable representation of the object.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, Profile):
        if additional_arg is not None:
            return obj.to_dict(additional_arg)
        else:
            return obj.to_dict()
    if isinstance(obj, dict):
        return {k: to_serializable(v, additional_arg) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v, additional_arg) for v in obj]
    if isinstance(obj, MpcStrategy):
        return obj.name
    return obj


class Med3paRecord:
    """
    Class to store and manage results from the MED3PA method on one set.
    """

    def __init__(self) -> None:
        self.metrics_by_dr: Dict[int, Dict] = {}
        self.models_evaluation: Dict[str, Dict] = {}
        self.profiles_manager: ProfilesManager = None
        self.datasets: Dict[str, MaskedDataset] = {}
        self.experiment_config = {}
        self.tree = None
        self.ipc_scores = None
        self.apc_scores = None
        self.mpc_scores = None

    def set_metrics_by_dr(self, metrics_by_dr: Dict) -> None:
        """
        Sets the calculated metrics by declaration rate.
        Args:
            metrics_by_dr (Dict): Dictionary of metrics by declaration rate.
        """
        self.metrics_by_dr = metrics_by_dr

    def set_profiles_manager(self, profile_manager: ProfilesManager) -> None:
        """
        Sets the profile manager for this Med3paResults instance.

        Args:
            profile_manager (ProfilesManager): The ProfileManager instance.
        """
        self.profiles_manager = profile_manager

    def set_models_evaluation(self, ipc_evaluation: Dict, apc_evaluation: Dict = None,
                              mpc_evaluation: Dict = None) -> None:
        """
        Sets models evaluation metrics.
        Args:
            ipc_evaluation (Dict): Evaluation metrics for IPC model.
            apc_evaluation (Dict): Evaluation metrics for APC model.
            mpc_evaluation (Dict): Evaluation metrics for MPC model.
        """
        self.models_evaluation['IPC_evaluation'] = ipc_evaluation

        if apc_evaluation is not None:
            self.models_evaluation['APC_evaluation'] = apc_evaluation

        if mpc_evaluation is not None:
            self.models_evaluation['MPC_evaluation'] = mpc_evaluation

    def set_tree(self, tree: TreeRepresentation) -> None:
        """
        Sets the constructed tree.
        """
        self.tree = tree

    def set_dataset(self, mode: str, dataset: MaskedDataset) -> None:
        """
        Saves the dataset for a given sample ratio.
        Args:
            mode (str): The modality of dataset, either 'ipc', 'apc', or 'mpc'.
            dataset (MaskedDataset): The MaskedDataset instance.
        """

        self.datasets[mode] = dataset

    def save(self, file_path: str) -> None:
        """
        Saves the experiment results.
        Args:
            file_path (str): The file path to save the JSON files.
        """
        # Ensure the main directory exists
        os.makedirs(file_path, exist_ok=True)

        metrics_file_path = os.path.join(file_path, 'metrics_dr.json')
        with open(metrics_file_path, 'w') as file:
            json.dump(self.metrics_by_dr, file, default=to_serializable, indent=4)

        if self.profiles_manager is not None:
            profiles_file_path = os.path.join(file_path, 'profiles.json')
            with open(profiles_file_path, 'w') as file:
                json.dump(self.profiles_manager.get_profiles(), file, default=to_serializable, indent=4)

            lost_profiles_file_path = os.path.join(file_path, 'lost_profiles.json')
            with open(lost_profiles_file_path, 'w') as file:
                json.dump(self.profiles_manager.get_lost_profiles(), file,
                          default=lambda x: to_serializable(x), indent=4)

        if self.models_evaluation is not None:
            models_evaluation_file_path = os.path.join(file_path, 'models_evaluation.json')
            with open(models_evaluation_file_path, 'w') as file:
                json.dump(self.models_evaluation, file, default=to_serializable, indent=4)

        for mode, dataset in self.datasets.items():
            dataset_path = os.path.join(file_path, f'dataset_{mode}.csv')
            dataset.save_to_csv(dataset_path)

        if self.tree is not None:
            tree_path = os.path.join(file_path, 'tree.json')
            self.tree.save_tree(tree_path)

    def save_to_dict(self) -> dict:
        """
        Collects the experiment results in a dictionary.
        Returns:
            dict: A dictionary containing all the saved elements.
        """
        result = {}

        # Store profiles if available
        if self.profiles_manager is not None:
            result['lost_profiles'] = to_serializable(self.profiles_manager.get_lost_profiles())
            result['profiles'] = to_serializable(self.profiles_manager.get_profiles())

        # Store metrics by declaration rate (DR)
        result['metrics_dr'] = self.metrics_by_dr

        # Store models evaluation if available
        if self.models_evaluation is not None:
            result['models_evaluation'] = self.models_evaluation

        # Store tree structure if available
        if self.tree is not None:
            result['tree'] = self.tree.to_dict()

        return result

    def get_profiles_manager(self) -> ProfilesManager:
        """
        Retrieves the profiles manager for this Med3paResults instance
        """
        return self.profiles_manager


class Med3paResults:
    """
    Class to store and manage results from the MED3PA complete experiment.
    """

    def __init__(self, test_record: Med3paRecord) -> None:
        """
        Initializes the Med3paResults class.

        Args:
            test_record: The test record for the MED3pa experiment.
        """
        self.test_record = test_record
        self.experiment_config = {}
        self.confidence_model = None

    def set_experiment_config(self, config: Dict[str, Any]) -> None:
        """
        Sets or updates the configuration for the MED3pa experiment.
        Args:
            config (Dict[str, Any]): A dictionary of experiment configuration.
        """
        self.experiment_config.update(config)

    def set_model(self, pc_model: IPCModel | APCModel | MPCModel) -> None:
        """
        Sets the confidence model for the Med3pa experiment.

        Args:
            pc_model (IPCModel): The confidence model for confidence predictions.
        """
        self.confidence_model = pc_model

    def save(self, file_path: str, save_med3paResults: bool = True) -> None:
        """
        Saves the experiment results.
        Args:
            file_path (str): The file path to save the JSON files.
            save_med3paResults (bool): Whether to save the results in a Med3paResults file. Defaults to True.
        """
        results = {}
        # Ensure the main directory exists
        os.makedirs(file_path, exist_ok=True)

        test_path = os.path.join(file_path, 'test')

        self.test_record.save(file_path=test_path)
        results['test'] = self.test_record.save_to_dict()

        experiment_config_path = os.path.join(file_path, 'experiment_config.json')
        with open(experiment_config_path, 'w') as file:
            json.dump(self.experiment_config, file, default=to_serializable, indent=4)

        results['infoConfig'] = {'experiment_config': self.experiment_config}

        if save_med3paResults:
            self.__generate_Med3paResults_from_dict(results, file_path=file_path)

    def save_models(self, file_path: str, file_id: str = "") -> None:
        """
        Saves the experiment confidence models as .pkl files, alongside the tree structure for the test set.
        Args:
            file_path (str): The file path to save the pickled files.
            file_id (str): Optional identifier to append to the filenames.
        """
        # Ensure the main directory exists
        os.makedirs(file_path, exist_ok=True)

        self.confidence_model.save_model(file_path=f"{file_path}/{file_id}_")

        if self.test_record.tree:
            tree_structure_path = os.path.join(file_path, f"{file_path}/{file_id}_tree.json")
            self.test_record.tree.save_tree(tree_structure_path)

    def __generate_Med3paResults_from_dict(self, data: dict, file_path: str) -> None:
        """
        Generates a Med3paResult file from the provided data dictionary then saves it in file_path with current time.
        Args:
            data (dict): A dictionary containing all the relevant data.
            file_path (str): Path where to save the Med3paResult file.

        """
        file_name = f"/MED3paResults_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}".replace(
            r'[^a-zA-Z0-9-_]', "")
        file_content = {"loadedFiles": {}, "isDetectron": False}

        # Process data based on tabs
        tabs = ["infoConfig", "test"]
        for tab in tabs:
            if tab in data:
                file_content["loadedFiles"][tab] = data[tab]
            else:
                print(f"Tab {tab} not found in")

        self.__save_dict_to_file(file_content, file_path + file_name + '.MED3paResults')

    @staticmethod
    def __to_string(value: Any) -> str:
        """
        Converts Any object to a string.

        Args:
            value (Any): Value to be converted to string.

        Returns:
            str: String representation of the value.
        """
        if value is None:
            return 'null'
        return str(value)

    def __write_list(self, file: TextIO, l: list, indent: int, inc: int) -> None:
        """
        Writes a list to the file.

        Args:
            file (TextIO): The file to write the dictionary to.
            l (list): The dictionary to be written in the file.
            indent (int): The indent level applied to the file.
            inc (int): The increments applied to the indent in the file.
        """
        if len(l) == 0:
            file.write("[]")
        else:
            file.write('[\n')
            for idx, item in enumerate(l):
                coma_list = '' if idx == len(l) - 1 else ','
                if isinstance(item, list):
                    self.__write_list(file, item, indent + inc, inc)
                elif isinstance(item, dict):
                    file.write(" " * (indent + inc))
                    self.__write_dict(file, item, indent + inc, inc)
                else:
                    file.write(' ' * (indent + inc) + '"' + self.__to_string(item) + '"')
                file.write(coma_list + "\n")
            file.write(' ' * indent + ']')

    def __write_dict(self, file: TextIO, d: dict, indent: int = 0, inc: int = 2) -> None:
        """
        Writes a dictionary to the file.

        Args:
            file (TextIO): The file to write the dictionary to.
            d (dict): The dictionary to be written in the file.
            indent (int): The indent level applied to the file.
            inc (int): The increments applied to the indent in the file.
        """
        if len(d) == 0:
            file.write("{}")
        else:
            file.write("{\n")
            for index, (key, value) in enumerate(d.items()):
                file.write(f'{" " * (indent + inc)}"{key}": ')
                coma = '' if index == len(d) - 1 else ','
                if isinstance(value, dict):
                    self.__write_dict(file, value, (indent + inc), inc)
                    file.write(coma + '\n')

                elif isinstance(value, list):
                    self.__write_list(file, value, (indent + inc), inc)
                    file.write(coma + '\n')

                elif isinstance(value, bool):
                    file.write(f'{str(value).lower()}{coma}\n')
                elif isinstance(value, str):
                    file.write(f'"{value}"{coma}\n')
                else:
                    file.write(f'{self.__to_string(value)}{coma}\n')
            file.write(" " * indent + "}")

    def __save_dict_to_file(self, dictionary: dict, file_path: str) -> None:
        """
        Saves a dictionary to a file.

        Args:
            dictionary (dict): The dictionary to be saved into the file.
            file_path (str): Path to the file to save the dictionary to.

        Raises:
            Exception: If an error occurs while saving the dictionary.
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                self.__write_dict(file, dictionary)

            print(f"Dictionary successfully saved to {file_path}")
        except Exception as e:
            print(f"Error saving dictionary to {file_path}: {e}")
