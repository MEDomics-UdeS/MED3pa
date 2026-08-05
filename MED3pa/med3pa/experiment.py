"""
Orchestrates the execution of the MED3pa method and integrates the functionality of other modules to run comprehensive experiments.
It includes ``Med3paExperiment`` to manage experiments.
"""
try:
    from checkpointer import checkpoint
except Exception:
    # fallback decorator that does nothing
    def checkpoint(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
from typing import Optional, Tuple, Union

from MED3pa.datasets import DatasetsManager, MaskedDataset
from MED3pa.med3pa.mdr import MDRCalculator
from MED3pa.med3pa.models import APCModel, IPCModel, MPCModel, MpcStrategy
from MED3pa.med3pa.profiles import ProfilesManager
from MED3pa.med3pa.results import Med3paResults, Med3paRecord
from MED3pa.med3pa.tree import TreeRepresentation
from MED3pa.med3pa.uncertainty import *
from MED3pa.models.base import BaseModelManager
from MED3pa.models.classification_metrics import *
from MED3pa.models.concrete_regressors import *


class Med3paExperiment:
    """
    Class to run the MED3PA method experiment.
    """

    @staticmethod
    @checkpoint(root_path="checkpoints", verbosity=False)
    def run(datasets_manager: DatasetsManager,
            base_model_manager: Optional[BaseModelManager] = None,
            uncertainty_metric: str = 'absolute_error',
            ipc_type: str = 'RandomForestRegressor',
            ipc_params: Optional[Dict] = None,
            ipc_grid_params: Optional[Dict] = None,
            ipc_cv: int = 4,
            pretrained_ipc: Optional[str] = None,
            apc_params: Optional[Dict] = None,
            apc_grid_params: Optional[Dict] = None,
            apc_cv: int = 4,
            pretrained_apc: Optional[str] = None,
            samples_ratio_min: int = 0,
            samples_ratio_max: int = 50,
            samples_ratio_step: int = 5,
            metrics_list: Optional[List[str]] = None,
            evaluate_models: bool = False,
            mode: str = 'mpc',
            mdr_size: int | float = 0.5,
            models_metrics: Optional[List[str]] = None,
            mpc_strategy="minimum",
            random_state: Optional[int] = None) -> Med3paResults:

        """
        Runs the MED3PA experiment on the testing set.

        Args:
            datasets_manager (DatasetsManager): the datasets manager containing the dataset to use in the experiment.
            base_model_manager (BaseModelManager, optional): Instance of BaseModelManager to get the base model,
                by default None.
            uncertainty_metric (str, optional): the uncertainty metric ysed to calculate uncertainty,
                by default absolute_error.
            ipc_type (str, optional): The regressor model to use for IPC, by default RandomForestRegressor.
            ipc_params (dict, optional): Parameters for initializing the IPC regressor model, by default None.
            ipc_grid_params (dict, optional): Grid search parameters for optimizing the IPC model, by default None.
            ipc_cv (int, optional): Number of cross-validation folds for optimizing the IPC model, by default None.
            pretrained_ipc (str, optional): path to a pretrained ipc, by default None.
            apc_params (dict, optional): Parameters for initializing the APC regressor model, by default None.
            apc_grid_params (dict, optional): Grid search parameters for optimizing the APC model, by default None.
            apc_cv (int, optional): Number of cross-validation folds for optimizing the APC model, by default None.
            pretrained_apc (str, optional): path to a pretrained apc, by default None.
            samples_ratio_min (int, optional): Minimum sample ratio, by default 0.
            samples_ratio_max (int, optional): Maximum sample ratio, by default 50.
            samples_ratio_step (int, optional): Step size for sample ratio, by default 5.
            metrics_list (list of str, optional): List of metrics to calculate, by default, multiple metrics included.
            evaluate_models (bool, optional): Whether to evaluate the models, by default False.
            mode (str): The modality of dataset, either 'ipc', 'apc', or 'mpc'.
            models_metrics (list of str, optional): List of metrics for model evaluation,
                by default ['MSE', 'RMSE', 'MAE'].
            mdr_size (int | float): The size or proportion of data to use for MDR curves evaluation, by default 0.5.
            mpc_strategy (str): The strategy to use to aggregate IPC and APC predictions in the MPC model, by default "minimum".
            random_state (int): The random state to use to split the data and train the models, by default None.


        Returns:
            Med3paResults: the results of the MED3PA experiment on the testing set.
        """
        # Verifications
        pretrained_apc, metrics_list, models_metrics = Med3paExperiment._verify_parameters(
            pretrained_ipc=pretrained_ipc,
            pretrained_apc=pretrained_apc,
            mode=mode,
            metrics_list=metrics_list,
            models_metrics=models_metrics)

        results_testing, pc_model = Med3paExperiment._run_by_set(datasets_manager=datasets_manager,
                                                                 base_model_manager=base_model_manager,
                                                                 uncertainty_metric=uncertainty_metric,
                                                                 ipc_type=ipc_type,
                                                                 ipc_params=ipc_params,
                                                                 ipc_grid_params=ipc_grid_params,
                                                                 ipc_cv=ipc_cv,
                                                                 pretrained_ipc=pretrained_ipc,
                                                                 apc_params=apc_params,
                                                                 apc_grid_params=apc_grid_params,
                                                                 apc_cv=apc_cv,
                                                                 pretrained_apc=pretrained_apc,
                                                                 samples_ratio_min=samples_ratio_min,
                                                                 samples_ratio_max=samples_ratio_max,
                                                                 samples_ratio_step=samples_ratio_step,
                                                                 metrics_list=metrics_list,
                                                                 evaluate_models=evaluate_models,
                                                                 models_metrics=models_metrics,
                                                                 mode=mode,
                                                                 mdr_size=mdr_size,
                                                                 mpc_strategy=mpc_strategy,
                                                                 random_state=random_state)

        results = Med3paResults(results_testing)
        med3pa_params = {
            'uncertainty_metric': uncertainty_metric,
            'samples_ratio_min': samples_ratio_min,
            'samples_ratio_max': samples_ratio_max,
            'samples_ratio_step': samples_ratio_step,
            'metrics_list': metrics_list,
            'evaluate_models': evaluate_models,
            'models_evaluation_metrics': models_metrics,
            'mode': mode,
            'pc_model': pc_model.get_info(),
        }
        experiment_config = {
            'experiment_name': "Med3paExperiment",
            'datasets': datasets_manager.get_info(),
            'base_model': base_model_manager.get_info() if base_model_manager is not None else None,
            'med3pa_params': med3pa_params
        }
        results.set_experiment_config(experiment_config)
        results.set_model(pc_model)
        return results

    @staticmethod
    def _run_by_set(datasets_manager: DatasetsManager,
                    base_model_manager: Optional[BaseModelManager],
                    uncertainty_metric: str,
                    ipc_type: str,
                    ipc_params: Optional[Dict],
                    ipc_grid_params: Optional[Dict],
                    ipc_cv: int,
                    pretrained_ipc: Optional[IPCModel | str],
                    apc_params: Optional[Dict],
                    apc_grid_params: Optional[Dict],
                    apc_cv: int,
                    pretrained_apc: Optional[APCModel | str],
                    samples_ratio_min: int,
                    samples_ratio_max: int,
                    samples_ratio_step: int,
                    metrics_list: List[str],
                    evaluate_models: bool,
                    mode: str,
                    models_metrics: List[str],
                    mdr_size: int | float,
                    random_state: Optional[int],
                    mpc_strategy: str | MpcStrategy) -> Tuple[Med3paRecord, IPCModel | APCModel | MPCModel]:
        """
        Orchestrates the MED3PA experiment on one specific set of the dataset.

        Args:
            datasets_manager (DatasetsManager): The datasets manager containing the dataset to use in the experiment.
            base_model_manager (BaseModelManager, optional): Instance of BaseModelManager to get the base model,
                by default None.
            uncertainty_metric (str, optional): The uncertainty metric used to calculate uncertainty,
                by default absolute_error.
            ipc_type (str, optional): The regressor model to use for IPC, by default RandomForestRegressor.
            ipc_params (dict, optional): Parameters for initializing the IPC regressor model, by default None.
            ipc_grid_params (dict, optional): Grid search parameters for optimizing the IPC model, by default None.
            ipc_cv (int, optional): Number of cross-validation folds for optimizing the IPC model, by default None.
            pretrained_ipc (str, optional): Path to a pretrained IPC model or an instance of IPCModel, by default None.
            apc_params (dict, optional): Parameters for initializing the APC regressor model, by default None.
            apc_grid_params (dict, optional): Grid search parameters for optimizing the APC model, by default None.
            apc_cv (int, optional): Number of cross-validation folds for optimizing the APC model, by default None.
            pretrained_apc (str, optional): Path to a pretrained APC model or an instance of APCModel, by default None.
            samples_ratio_min (int, optional): Minimum sample ratio, by default 0.
            samples_ratio_max (int, optional): Maximum sample ratio, by default 50.
            samples_ratio_step (int, optional): Step size for sample ratio, by default 5.
            metrics_list (list of str, optional): List of metrics to calculate.
            evaluate_models (bool, optional): Whether to evaluate the models, by default False.
            mode (str): The modality of dataset, either 'ipc', 'apc', or 'mpc'.
            models_metrics (list of str, optional): List of metrics for model evaluation.

            mdr_size (int or float): The sample size reserved to calculate the MDR curves, by default 50%.
            random_state (int, optional): The random state to use, by default None.
            mpc_strategy (str): The strategy to use to combine IPC and APC predictions in the MPC.

        Returns:
            Med3paRecord: The results of the MED3PA experiment.
            IPCModel | APCModel | MPCModel : The trained or loaded models.
        """

        # Step 1 : datasets and base model setting
        dataset, features = Med3paExperiment._setup_dataset(datasets_manager=datasets_manager,
                                                            base_model_manager=base_model_manager)

        # Step 2 : Calculate uncertainty values
        if base_model_manager is not None:
            threshold = base_model_manager.threshold
        else:
            threshold = 0.5
        uncertainty_calc = UncertaintyCalculator(uncertainty_metric, threshold=threshold)

        # Step 3: Set up splits to evaluate the confidence models
        dataset_pc, dataset_mdr, dataset_evaluate = Med3paExperiment._setup_splits(
            dataset=dataset,
            evaluate_models=evaluate_models,
            pretrained_ipc=pretrained_ipc,
            pretrained_apc=pretrained_apc,
            mode=mode,
            mdr_size=mdr_size,
            random_state=random_state
            )

        # Step 4: Create and train models
        IPC_model, APC_model, MPC_model = Med3paExperiment._train_models(
            dataset_pc=dataset_pc,
            features=features,
            ipc_type=ipc_type,
            ipc_params=ipc_params,
            ipc_grid_params=ipc_grid_params,
            ipc_cv=ipc_cv,
            pretrained_ipc=pretrained_ipc,
            apc_params=apc_params,
            apc_grid_params=apc_grid_params,
            apc_cv=apc_cv,
            pretrained_apc=pretrained_apc,
            mpc_strategy=mpc_strategy,
            uncertainty_calc=uncertainty_calc,
        random_state=random_state)

        results = Med3paRecord()

        # Save the tree structure created by the APCModel
        tree = APC_model.treeRepresentation
        results.set_tree(tree=tree)

        # Step 5: Calculate profiles and their metrics
        confidence_model = IPC_model if mode == "ipc" else (APC_model if mode == "apc" else MPC_model)

        Med3paExperiment._calculate_profiles(
            results=results,
            dataset_mdr=dataset_mdr,
            features=features,
            tree=tree,
            confidence_model=confidence_model,
            samples_ratio_min=samples_ratio_min,
            samples_ratio_max=samples_ratio_max,
            samples_ratio_step=samples_ratio_step,
            metrics_list=metrics_list
        )

        # Step 6: Evaluate models if required
        Med3paExperiment._evaluate_models(
            results=results,
            IPC_model=IPC_model,
            APC_model=APC_model,
            MPC_model=MPC_model,
            dataset_evaluate=dataset_evaluate,
            uncertainty_calc=uncertainty_calc,
            evaluate_models=evaluate_models,
            models_metrics=models_metrics
        )

        if mode == "ipc":
            return results, IPC_model
        elif mode == "apc":
            return results, APC_model
        return results, MPC_model

    @staticmethod
    def _verify_parameters(pretrained_ipc: Optional[str | IPCModel],
                             pretrained_apc: Optional[APCModel | str],
                             mode: str, metrics_list: Optional[List[str]], models_metrics: Optional[List[str]]
                           ) -> Tuple[Optional[APCModel | str], List[str], List[str]]:
        """
        Verifies the parameters passed to the experiment.

        Args:
            pretrained_ipc (str | IPCModel): Path to a pretrained IPC model or an instance of IPCModel.
            pretrained_apc (str, APCModel): Path to a pretrained APC model or an instance of APCModel, by default None.
            mode (str): The modality of the confidence model, either 'ipc', 'apc', or 'mpc'.
            metrics_list (List[str]): List of metrics to calculate.
        """
        if not pretrained_ipc and pretrained_apc:
            print("Pretrained instance is ignored since IPC is not pretrained")
            pretrained_apc = None

        valid_modes = ['mpc', 'apc', 'ipc']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. The mode must be one of {valid_modes}.")

        if not metrics_list:
            metrics_list = ClassificationEvaluationMetrics.supported_metrics()

        if not models_metrics:
            models_metrics = RegressionEvaluationMetrics.supported_metrics()

        return pretrained_apc, metrics_list, models_metrics

    @staticmethod
    def _setup_dataset(datasets_manager: DatasetsManager,
                        base_model_manager: Optional[BaseModelManager]) -> Tuple[MaskedDataset, List[str]]:
        """
        Sets up the dataset for the experiment.

        Args:
            datasets_manager (DatasetsManager): The datasets manager containing the dataset to use in the experiment.
            base_model_manager (BaseModelManager, optional): Instance of BaseModelManager to get the base model.

        Returns:
            Tuple[MaskedDataset, List[str]]: The dataset and features.
        """
        # retrieve different dataset components needed for the experiment
        features = datasets_manager.get_column_labels()
        dataset = datasets_manager.get_dataset_by_type(dataset_type="testing", return_instance=True)

        # Initialize predict probabilities if not provided
        if dataset.get_pseudo_probabilities() is None:
            if base_model_manager is None:
                raise ValueError("Either the base model or the predicted probabilities should be provided!")
            predicted_probabilities = base_model_manager.predict_proba(dataset.get_observations())[:, 1]
            threshold = base_model_manager.threshold
            dataset.set_pseudo_probs_labels(predicted_probabilities, threshold)
        
        return dataset, features

    @staticmethod
    def _setup_splits(dataset: MaskedDataset,
                      evaluate_models: bool,
                      pretrained_ipc: Optional[str | IPCModel],
                      pretrained_apc: Optional[APCModel | str],
                      mode: str,
                      mdr_size: int | float,
                      random_state: Optional[int]) -> Tuple[MaskedDataset|None, MaskedDataset, MaskedDataset|None]:
        """
        Sets up the data splits for the experiment.

        Args:
            dataset (MaskedDataset): The dataset to split.
            evaluate_models (bool): Whether to evaluate the models.
            pretrained_ipc (str | IPCModel): Path to a pretrained IPC model or an instance of IPCModel.
            random_state (int, optional): The random state to use.

        Returns:
            Tuple[MaskedDataset, MaskedDataset,MaskedDataset]: The training dataset, mdr dataset, and
             evaluation dataset if applicable.
        """
        if evaluate_models:
            dataset_train_complete, dataset_evaluate = dataset.train_test_split(test_size=0.1,
                                                                                random_state=random_state)
        else:
            dataset_train_complete = dataset.clone()
            dataset_evaluate = None

        # Split the data if pretrained models are not available
        if pretrained_ipc and (pretrained_apc or mode == "ipc"):
            # If we reuse both IPC and APC models, no training required, all data is used for MDR curve generation
            dataset_pc = None
            dataset_mdr = dataset_train_complete

        else:
            # If we train either APC or IPC, split data in half: one half for IPC and APC, other half for MDR curves
            dataset_pc, dataset_mdr = dataset_train_complete.train_test_split(test_size=mdr_size,
                                                                              random_state=random_state)

        
        return dataset_pc, dataset_mdr, dataset_evaluate

    @staticmethod
    def _train_models(dataset_pc: Optional[MaskedDataset],
                       features: List[str],
                       ipc_type: str,
                       ipc_params: Optional[Dict],
                       ipc_grid_params: Optional[Dict],
                       ipc_cv: int,
                       pretrained_ipc: Optional[str | IPCModel],
                       apc_params: Optional[Dict],
                       apc_grid_params: Optional[Dict],
                       apc_cv: int,
                       pretrained_apc: Optional[APCModel | str],
                      mpc_strategy: str | MpcStrategy,
                      uncertainty_calc: UncertaintyCalculator,
                      random_state: Optional[int] = None) -> Tuple[IPCModel, APCModel, MPCModel]:
        """
        Trains the IPC and APC models.

        Args:
            dataset_pc (MaskedDataset): The dataset to train on.
            features (List[str]): The features of the dataset.
            ipc_type (str): The type of IPC model to use.
            ipc_params (Dict): Parameters for initializing the IPC model.
            ipc_grid_params (Dict): Grid search parameters for optimizing the IPC model.
            ipc_cv (int): Number of cross-validation folds for optimizing the IPC model.
            pretrained_ipc (str): Path to a pretrained IPC model or an instance of IPCModel.
            apc_params (Dict): Parameters for initializing the APC model.
            apc_grid_params (Dict): Grid search parameters for optimizing the APC model.
            apc_cv (int): Number of cross-validation folds for optimizing the APC model.
            pretrained_apc (str | APCModel): Path to a pretrained APC model or an instance of APCModel, by default None.

        Returns:
            Tuple[IPCModel, APCModel, MPCModel]: The trained or loaded IPC, APC and MPC models.
        """
        # Create and train IPCModel
        if not pretrained_ipc:
            IPC_model = IPCModel(model_name=ipc_type, params=ipc_params, pretrained_model=None,
                                 random_state=random_state)
            uncertainty_ipc = uncertainty_calc.calculate_uncertainty(x=dataset_pc.get_observations(),
                                                                             predicted_prob=dataset_pc.get_pseudo_probabilities(),
                                                                             y_true=dataset_pc.get_true_labels())
            if ipc_type == 'EnsembleRandomForestRegressor':
                # Add class weight correction to train the EnsembleRandomForestRegressor
                class_1_prop = int(np.sum(dataset_pc.get_true_labels())) / len(dataset_pc.get_true_labels())
                sample_weight = np.where(dataset_pc.get_true_labels() == 0,
                                         1 / (1 - class_1_prop),
                                         1 / class_1_prop)
            else:
                sample_weight = None

            # Optimize IPC model if grid params were provided
            if ipc_grid_params is not None and len(dataset_pc.get_true_labels()) > 4:
                # No optimization if 4 or less samples
                IPC_model.optimize(param_grid=ipc_grid_params, cv=ipc_cv,
                                   x=dataset_pc.get_observations(),
                                   confidence_score=uncertainty_ipc,
                                   sample_weight=sample_weight)
                print("IPC Model optimization complete.")
            else:
                IPC_model.train(dataset_pc.get_observations(), uncertainty_ipc, sample_weight=sample_weight)
                print("IPC Model training complete.")

        elif isinstance(pretrained_ipc, str):
            IPC_model = IPCModel(model_name=ipc_type, params=ipc_params, pretrained_model=pretrained_ipc)
            print("Loaded a pretrained IPC model.")
        else:
            IPC_model = pretrained_ipc
            print("Used a trained IPC instance.")

        # Create and train APCModel
        if pretrained_apc and pretrained_ipc:
            # If IPC and APC are pretrained
            if isinstance(pretrained_apc, str):
                APC_model = APCModel(features=features, params=apc_params, pretrained_model=pretrained_apc)
                print("Loaded a pretrained APC model.")
            else:
                APC_model = pretrained_apc
                print("Used a trained APC instance.")

        else:
            # Predict IPC values
            IPC_predictions_apc = IPC_model.predict(dataset_pc.get_observations())
            # Train APC model
            APC_model = APCModel(features=features, params=apc_params, random_state=random_state)

            if apc_grid_params is not None and len(IPC_predictions_apc) > 4:
                APC_model.optimize(apc_grid_params, apc_cv, dataset_pc.get_observations(), IPC_predictions_apc)
                print("APC Model optimization complete.")
            else:
                APC_model.train(dataset_pc.get_observations(), IPC_predictions_apc)
                print("APC Model training complete.")

        # Create MPC model
        MPC_model = MPCModel(IPC_model=IPC_model, APC_model=APC_model, strategy=mpc_strategy)

        return IPC_model, APC_model, MPC_model

    @staticmethod
    def _calculate_profiles(results: Med3paRecord,
                              dataset_mdr: MaskedDataset,
                              features: List[str],
                              tree: TreeRepresentation,
                              confidence_model: IPCModel | APCModel | MPCModel,
                              samples_ratio_min: int,
                              samples_ratio_max: int,
                              samples_ratio_step: int,
                              metrics_list: List[str]):
        """
        Calculates the profiles and their metrics.

        Args:
            results (Med3paRecord): The results record.
            dataset_mdr (MaskedDataset): The MDR dataset.
            features (List[str]): The features of the dataset.
            tree (any): The tree representation.
            confidence_model (IPCModel | APCModel | MPCModel): The confidence model.
            samples_ratio_min (int): Minimum sample ratio.
            samples_ratio_max (int): Maximum sample ratio.
            samples_ratio_step (int): Step size for sample ratio.
            metrics_list (List[str]): List of metrics to calculate.
        """
        profiles_manager = ProfilesManager(features)

        # get confidence scores
        confidence_preds = confidence_model.predict(dataset_mdr.get_observations())

        # get confidence scores
        metrics_by_dr = MDRCalculator.calc_metrics_by_dr(dataset=dataset_mdr,
                                                         confidence_scores=confidence_preds,
                                                         metrics_list=metrics_list)
        results.set_metrics_by_dr(metrics_by_dr)

        for samples_ratio in range(samples_ratio_min, samples_ratio_max + 1, samples_ratio_step):
            # Calculate profiles and their metrics by declaration rate
            MDRCalculator.calc_profiles(profiles_manager=profiles_manager, tree=tree, dataset=dataset_mdr,
                                        features=features, confidence_scores=confidence_preds,
                                        min_samples_ratio=samples_ratio, metrics_list=metrics_list)
            print("Results extracted for minimum_samples_ratio = ", samples_ratio)
        results.set_profiles_manager(deepcopy(profiles_manager))

    @staticmethod
    def _evaluate_models(results: Med3paRecord,
                         IPC_model: IPCModel,
                         APC_model: APCModel,
                         MPC_model: MPCModel,
                         dataset_evaluate: Optional[MaskedDataset],
                         uncertainty_calc: UncertaintyCalculator,
                         evaluate_models: bool,
                         models_metrics: List[str]):
        """
        Evaluates the models.

        Args:
            results (Med3paRecord): The results record.
            IPC_model (IPCModel): The trained or loaded IPC model.
            APC_model (APCModel): The trained or loaded APC model.
            MPC_model (MPCModel): The created MPC model.
            dataset_evaluate (MaskedDataset): The evaluation dataset.
            uncertainty_calc (UncertaintyCalculator): The uncertainty calculator.
            evaluate_models (bool): Whether to evaluate the models.
            models_metrics (List[str]): List of metrics for model evaluation.
        """
        if evaluate_models:
            if not isinstance(dataset_evaluate, MaskedDataset):
                raise ValueError(f"Wrong data input to evaluate confidence models: {type(dataset_evaluate)}")
            uncertainty_evaluate = uncertainty_calc.calculate_uncertainty(x=dataset_evaluate.get_observations(),
                                                                          predicted_prob=dataset_evaluate.get_pseudo_probabilities(),
                                                                          y_true=dataset_evaluate.get_true_labels())
            IPC_evaluation = IPC_model.evaluate(X=dataset_evaluate.get_observations(), y=uncertainty_evaluate,
                                                eval_metrics=models_metrics)
            APC_evaluation = APC_model.evaluate(X=dataset_evaluate.get_observations(), y=uncertainty_evaluate,
                                                eval_metrics=models_metrics)
            MPC_evaluation = MPC_model.evaluate(X=dataset_evaluate.get_observations(), y=uncertainty_evaluate,
                                                eval_metrics=models_metrics)
            results.set_models_evaluation(IPC_evaluation, APC_evaluation, MPC_evaluation)