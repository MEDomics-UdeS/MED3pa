"""
Module: model_factories
This module provides factory classes for creating and managing machine learning models, 
specifically focusing on XGBoost models.
"""

import pickle
import json
import re
import xgboost as xgb
from .concrete_classifiers import Model, XGBoostModel
from .abstract_models import Model, ClassificationModel, RegressionModel


class ModelFactory:
    """
    A factory class for creating models with different types, using the factory design pattern.
    It supports creating models based on hyperparameters or loading them from pickled files.
    """
    
    model_mapping = {
        'XGBoostModel': [xgb.Booster, xgb.XGBClassifier],
    }

    factories = {
        'XGBoostModel': lambda: XGBoostFactory(),
    }

    @staticmethod
    def get_factory(model_type: str) -> 'ModelFactory':
        """
        Retrieves the factory object for the given model type.

        Parameters
        ----------
        model_type : str
            The type of model for which the factory is to be retrieved.

        Returns
        -------
        ModelFactory
            An instance of the factory associated with the given model type.

        Raises
        ------
        ValueError
            If no factory is available for the given model type.
        """
        factory_initializer = ModelFactory.factories.get(model_type)
        if factory_initializer:
            return factory_initializer()
        else:
            raise ValueError(f"No factory available for model type: {model_type}")
        
    @staticmethod
    def create_model_with_hyperparams(model_type: str, hyperparams: dict) -> Model:
        """
        Creates a model of the specified type with the given hyperparameters.

        Parameters
        ----------
        model_type : str
            The type of model to create.
        hyperparams : dict
            A dictionary of hyperparameters for the model.

        Returns
        -------
        Model
            A model instance of the specified type, initialized with the given hyperparameters.
        """
        factory = ModelFactory.get_factory(model_type)
        return factory.create_model_with_hyperparams(hyperparams)

    @staticmethod
    def create_model_from_pickled(pickled_file_path: str) -> Model:
        """
        Creates a model by loading it from a pickled file.

        Parameters
        ----------
        pickled_file_path : str
            The file path to the pickled model file.

        Returns
        -------
        Model
            A model instance loaded from the pickled file.

        Raises
        ------
        IOError
            If there is an error loading the model from the file.
        TypeError
            If the loaded model is not of a supported type.
        """
        try:
            with open(pickled_file_path, 'rb') as file:
                loaded_model = pickle.load(file)
        except Exception as e:
            raise IOError(f"Failed to load the model from {pickled_file_path}: {e}")
        
        for model_type, model_classes in ModelFactory.model_mapping.items():
            if any(isinstance(loaded_model, model_class) for model_class in model_classes):
                factory = ModelFactory.get_factory(model_type)
                return factory.create_model_from_pickled(loaded_model)

        raise TypeError("The loaded model is not of a supported type")


class XGBoostFactory(ModelFactory):
    """
    A factory for creating XGBoost model objects, either from hyperparameters or by loading from pickled files.
    Inherits from ModelFactory and specifies creation methods for XGBoost models.
    """
    
    def create_model_with_hyperparams(self, hyperparams: dict) -> XGBoostModel:
        """
        Creates an XGBoostModel with the given hyperparameters.

        Parameters
        ----------
        hyperparams : dict
            A dictionary of hyperparameters for the XGBoost model.

        Returns
        -------
        XGBoostModel
            An instance of XGBoostModel initialized with the given hyperparameters.
        """
        return XGBoostModel(params=hyperparams)

    def create_model_from_pickled(self, loaded_model: xgb.Booster | xgb.XGBClassifier) -> XGBoostModel:
        """
        Recreates an XGBoostModel from a loaded pickled model.

        Parameters
        ----------
        loaded_model : xgb.Booster | xgb.XGBClassifier
            The loaded model object, expected to be an instance of xgb.Booster or xgb.XGBClassifier.

        Returns
        -------
        XGBoostModel
            An instance of XGBoostModel created from the loaded model.

        Raises
        ------
        TypeError
            If the loaded model is not a supported implementation of the XGBoost model.
        ValueError
            If the XGBoost model version is not supported.
        """
        if isinstance(loaded_model, (xgb.Booster, xgb.XGBClassifier)):
            if self.check_version(loaded_model):
                extracted_params = self.extract_params(loaded_model)
                return XGBoostModel(params=extracted_params, model=loaded_model, model_class=type(loaded_model))
            else:
                raise ValueError("XGBoost model version is not supported. Please use version 2.0.0 or later.")
        else:
            raise TypeError("Loaded model is not an XGBoost model")

    def check_version(self, loaded_model: xgb.Booster | xgb.XGBClassifier) -> bool:
        """
        Checks the version of the loaded XGBoost model to ensure it is supported.

        Parameters
        ----------
        loaded_model : xgb.Booster | xgb.XGBClassifier
            The loaded model object.

        Returns
        -------
        bool
            True if the model version is supported, False otherwise.
        """
        config_json = loaded_model.save_config()
        config = json.loads(config_json)
        version_list = config.get('version', 'Not available')
        if isinstance(version_list, list):
            version_str = '.'.join(map(str, version_list))
        else:
            version_str = version_list

        print(version_str)
        version_match = re.match(r'(\d+)\.(\d+)\.(\d+)', version_str)
        if version_match:
            major, minor, patch = map(int, version_match.groups())
            return (major, minor, patch) >= (2, 0, 0)
        else:
            return False

    def extract_params(self, loaded_model: xgb.Booster | xgb.XGBClassifier) -> dict:
        """
        Extracts the parameters from a loaded XGBoost model.

        Parameters
        ----------
        loaded_model : xgb.Booster | xgb.XGBClassifier
            The loaded model object.

        Returns
        -------
        dict
            A dictionary of extracted parameters.
        """
        try:
            boosted_rounds = loaded_model.num_boosted_rounds()
            config_json = loaded_model.save_config()
            config = json.loads(config_json)
        except AttributeError as e:
            print(f"Error extracting configuration from model: {e}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON configuration: {e}")
            return {}

        try:
            learner = config['learner']
            gradient_booster = learner['gradient_booster']
            
            general_params = learner['generic_param']
            booster_params = gradient_booster.get('gbtree_train_param', {})
            tree_params = gradient_booster.get('tree_train_param', {})
            
            updater_params = {}
            if 'updater' in gradient_booster and isinstance(gradient_booster['updater'], list):
                for updater in gradient_booster['updater']:
                    if 'hist_train_param' in updater:
                        updater_params.update(updater['hist_train_param'])
            
            learning_task_params = learner['learner_train_param']
            objective_params = learner['objective'].get('reg_loss_param', {})
            learner_model_params = learner['learner_model_param']

            params = {}
            params.update(general_params)
            params.update(booster_params)
            params.update(tree_params)
            params.update(updater_params)
            params.update(learning_task_params)
            params.update(objective_params)
            params.update(learner_model_params)

            if 'metrics' in learner:
                metrics = [metric['name'] for metric in learner['metrics']]
                if metrics:
                    params['eval_metric'] = metrics
            params['num_boost_rounds'] = boosted_rounds

            for key, value in params.items():
                try:
                    if isinstance(value, str) and ('.' in value or 'E' in value or 'e' in value):
                        params[key] = float(value)
                    else:
                        params[key] = int(value)
                except (ValueError, TypeError):
                    pass

            removable_keys = ['num_trees']
            for key in removable_keys:
                params.pop(key, None)

        except KeyError as e:
            print(f"Key error while extracting parameters: {e}")
            return {}

        return params
