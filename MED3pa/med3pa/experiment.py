"""
Orchestrates the execution of the MED3pa method and integrates the functionality of other modules to run comprehensive experiments.
It includes ``Med3paExperiment`` to manage experiments.
"""

from checkpointer import checkpoint
from copy import deepcopy
from typing import Optional, Tuple, Union

from MED3pa.datasets import DatasetsManager, MaskedDataset
from MED3pa.med3pa.mdr import MDRCalculator
from MED3pa.med3pa.models import APCModel, IPCModel, MPCModel
from MED3pa.med3pa.profiles import ProfilesManager
from MED3pa.med3pa.results import Med3paResults, Med3paRecord
from MED3pa.med3pa.tree import TreeRepresentation
from MED3pa.med3pa.confidence import *
from MED3pa.models.base import BaseModelManager
from MED3pa.models.classification_metrics import *
from MED3pa.models.concrete_regressors import *
from MED3pa.uncertainty_explainer.explainer import Explainer


class Med3paExperiment:
    """
    Class to run the MED3PA method experiment.
    """

    @staticmethod
    @checkpoint(root_path="checkpoints", verbosity=False)
    def run(datasets_manager: DatasetsManager,
            base_model_manager: Optional[BaseModelManager] = None,
            confidence_metric: str = 'absolute_error',
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
            random_state: Optional[int] = None,
            explainer: bool|Explainer = False,
            pretrained_source_error_model: Optional[IPCModel | str] = None) -> Med3paResults:

        """
        Runs the MED3PA experiment on the testing set.

        Args:
            datasets_manager (DatasetsManager): the datasets manager containing the dataset to use in the experiment.
            base_model_manager (BaseModelManager, optional): Instance of BaseModelManager to get the base model,
                by default None.
            confidence_metric (str, optional): the confidence metric used to calculate the predictive confidence,
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
            explainer (bool | Explainer): If uncertainty sources need to be explained in each uncertainty profiles, by default False.
            pretrained_source_error_model (IPCModel | str, optional): Source error model of the explainer (an IPC
                model predicting the confidence on the reference dataset), as an IPCModel instance or a path to a
                pretrained IPC model of type `ipc_type`, e.g. the `source_error_model` of a previous run's results.
                By default None: it is trained on the reference dataset.


        Returns:
            Med3paResults: the results of the MED3PA experiment on the testing set.
        """
        # Verifications
        pretrained_apc, metrics_list, models_metrics, explainer = Med3paExperiment._verify_parameters(
            pretrained_ipc=pretrained_ipc,
            pretrained_apc=pretrained_apc,
            mode=mode,
            metrics_list=metrics_list,
            models_metrics=models_metrics,
            explainer=explainer,
            datasets_manager=datasets_manager,
            base_model_manager=base_model_manager)

        results_testing, pc_model, source_error_model = Med3paExperiment._run_by_set(
                                                                 datasets_manager=datasets_manager,
                                                                 base_model_manager=base_model_manager,
                                                                 confidence_metric=confidence_metric,
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
                                                                 random_state=random_state,
                                                                 explainer=explainer,
                                                                 pretrained_source_error_model=pretrained_source_error_model)

        results = Med3paResults(results_testing)
        med3pa_params = {
            'confidence_metric': confidence_metric,
            'samples_ratio_min': samples_ratio_min,
            'samples_ratio_max': samples_ratio_max,
            'samples_ratio_step': samples_ratio_step,
            'metrics_list': metrics_list,
            'evaluate_models': evaluate_models,
            'models_evaluation_metrics': models_metrics,
            'mode': mode,
            'pc_model': pc_model.get_info(),
            'source_error_model': source_error_model.get_info() if source_error_model is not None else None,
        }
        experiment_config = {
            'experiment_name': "Med3paExperiment",
            'datasets': datasets_manager.get_info(),
            'base_model': base_model_manager.get_info() if base_model_manager is not None else None,
            'med3pa_params': med3pa_params
        }
        results.set_experiment_config(experiment_config)
        results.set_model(pc_model)
        results.set_source_error_model(source_error_model)
        return results

    @staticmethod
    def _run_by_set(datasets_manager: DatasetsManager,
                    base_model_manager: Optional[BaseModelManager],
                    confidence_metric: str,
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
                    mpc_strategy: str,
                    explainer: bool | Explainer,
                    pretrained_source_error_model: Optional[IPCModel | str] = None
                    ) -> Tuple[Med3paRecord, IPCModel | APCModel | MPCModel, Optional[IPCModel]]:
        """
        Orchestrates the MED3PA experiment on one specific set of the dataset.

        Args:
            datasets_manager (DatasetsManager): The datasets manager containing the dataset to use in the experiment.
            base_model_manager (BaseModelManager, optional): Instance of BaseModelManager to get the base model,
                by default None.
            confidence_metric (str, optional): The confidence metric used to calculate the predictive confidence,
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
            explainer (bool | Explainer):
            pretrained_source_error_model (IPCModel | str, optional): Source error model of the explainer, as an
                IPCModel instance or a path to a pretrained IPC model, by default None (trained by the explainer).

        Returns:
            Med3paRecord: The results of the MED3PA experiment.
            IPCModel | APCModel | MPCModel : The trained or loaded models.
            IPCModel | None: The source error model of the explainer, None if its methods do not use one.
        """

        # Step 1 : datasets and base model setting
        dataset, features = Med3paExperiment._setup_dataset(datasets_manager=datasets_manager,
                                                            base_model_manager=base_model_manager)

        # Step 2 : Calculate confidence values
        confidence_calc = ConfidenceCalculator(confidence_metric)

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
            confidence_calc=confidence_calc,
        random_state=random_state)

        # the explainer's error models are the IPC model and the same IPC class trained on the reference dataset
        # (or the given pretrained source error model)
        if isinstance(pretrained_source_error_model, str):
            pretrained_source_error_model = IPCModel(model_name=ipc_type, params=ipc_params,
                                                     pretrained_model=pretrained_source_error_model)
            print("Loaded a pretrained source error model.")
        explainer.set_error_model(error_model=IPC_model, confidence_calc=confidence_calc,
                                  source_error_model=pretrained_source_error_model)

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
            metrics_list=metrics_list,
            explainer=explainer
        )

        # Step 5b: Additional trees of profiles: APC of the source error model, covariate and outcome shifts
        Med3paExperiment._calculate_additional_trees(
            results=results,
            dataset_mdr=dataset_mdr,
            features=features,
            APC_model=APC_model,
            confidence_model=confidence_model,
            samples_ratio_min=samples_ratio_min,
            samples_ratio_max=samples_ratio_max,
            samples_ratio_step=samples_ratio_step,
            metrics_list=metrics_list,
            explainer=explainer,
            random_state=random_state
        )

        # Step 6: Evaluate models if required
        Med3paExperiment._evaluate_models(
            results=results,
            IPC_model=IPC_model,
            APC_model=APC_model,
            MPC_model=MPC_model,
            dataset_evaluate=dataset_evaluate,
            confidence_calc=confidence_calc,
            evaluate_models=evaluate_models,
            models_metrics=models_metrics
        )

        source_error_model = explainer.get_source_error_model()

        if mode == "ipc":
            return results, IPC_model, source_error_model
        elif mode == "apc":
            return results, APC_model, source_error_model
        return results, MPC_model, source_error_model

    @staticmethod
    def _verify_parameters(pretrained_ipc: Optional[str | IPCModel],
                           pretrained_apc: Optional[APCModel | str],
                           mode: str, metrics_list: Optional[List[str]], models_metrics: Optional[List[str]],
                           explainer: bool|Explainer, datasets_manager: DatasetsManager,
                           base_model_manager: BaseModelManager
                           ) -> Tuple[Optional[APCModel | str], List[str], List[str], Explainer]:
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

        if explainer is True:
            explainer = Explainer()
        if isinstance(explainer, Explainer) and not explainer.context_is_set():
            # the reference (source) dataset of the explainers: the reference set, else the base model training set,
            # on which the base model losses are optimistic (biased source error models and shift tests)
            if datasets_manager.reference_set is not None:
                reference_dataset = datasets_manager.get_dataset_by_type(dataset_type="reference", return_instance=True)
            else:
                if explainer.methods_count() > 0:
                    print("Warning: no reference set in the DatasetsManager, the explainer uses the base model training "
                          "set as reference; its shift tests are biased by the base model's in-sample losses. Set a "
                          "held-out 'reference' dataset (same domain, not used to train the base model).")
                reference_dataset = datasets_manager.get_dataset_by_type(dataset_type="training", return_instance=True)
            f_model = base_model_manager.get_instance()
            explainer.set_context(reference_dataset=reference_dataset, f_model=f_model)
        if explainer is False:
            explainer = Explainer(methods=[])


        return pretrained_apc, metrics_list, models_metrics, explainer

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
        dataset_types = [dataset_type for dataset_type, dataset in (
            ("training", datasets_manager.base_model_training_set), ("reference", datasets_manager.reference_set))
                         if dataset is not None]
        dataset_types.append("testing")  # last: the returned dataset

        for dataset_type in dataset_types:
            dataset = datasets_manager.get_dataset_by_type(dataset_type=dataset_type, return_instance=True)

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
                      mpc_strategy: str,
                      confidence_calc: ConfidenceCalculator,
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
            confidence_ipc = confidence_calc.calculate_confidence(dataset_pc.get_observations(),
                                                                             dataset_pc.get_pseudo_probabilities(),
                                                                             dataset_pc.get_true_labels())
            if ipc_type == 'EnsembleRandomForestRegressor':
                # Add class weight correction to train the EnsembleRandomForestRegressor
                class_1_prop = int(np.sum(dataset_pc.get_true_labels())) / len(dataset_pc.get_true_labels())
                sample_weight = np.where(dataset_pc.get_true_labels() == 0,
                                         1 / (1 - class_1_prop),
                                         1 / class_1_prop)
            else:
                sample_weight = None
            if ipc_type == 'BalancedEnsembleRandomForestRegressor':
                # The BalancedEnsembleRandomForestRegressor computes its class and error weights from the labels
                train_params = {'y_true': dataset_pc.get_true_labels(), 'y_pred': dataset_pc.get_pseudo_labels()}
            else:
                train_params = {}

            # Optimize IPC model if grid params were provided
            if ipc_grid_params is not None and len(dataset_pc.get_true_labels()) > 4:
                # No optimization if 4 or less samples
                IPC_model.optimize(param_grid=ipc_grid_params, cv=ipc_cv,
                                   x=dataset_pc.get_observations(),
                                   confidence_score=confidence_ipc,
                                   sample_weight=sample_weight,
                                   **train_params)
                print("IPC Model optimization complete.")
            else:
                IPC_model.train(dataset_pc.get_observations(), confidence_ipc, sample_weight=sample_weight,
                                **train_params)
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
                            metrics_list: List[str],
                            explainer: Explainer):
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
        # the profiles are explained on dataset_mdr
        explainer.set_target_dataset(target_dataset=dataset_mdr)

        # get confidence scores
        confidence_preds = confidence_model.predict(dataset_mdr.get_observations())

        # get confidence scores
        metrics_by_dr = MDRCalculator.calc_metrics_by_dr(dataset=dataset_mdr,
                                                         confidence_scores=confidence_preds,
                                                         metrics_list=metrics_list)
        results.set_metrics_by_dr(metrics_by_dr)

        explanations_cache = {}  # each profile is explained once, for all the declaration rates and samples ratios
        for samples_ratio in range(samples_ratio_min, samples_ratio_max + 1, samples_ratio_step):
            # Calculate profiles and their metrics by declaration rate
            MDRCalculator.calc_profiles(profiles_manager=profiles_manager, tree=tree, dataset=dataset_mdr,
                                        features=features, confidence_scores=confidence_preds,
                                        min_samples_ratio=samples_ratio, metrics_list=metrics_list,
                                        explainer=explainer, explanations_cache=explanations_cache)
            print("Results extracted for minimum_samples_ratio = ", samples_ratio)
        results.set_profiles_manager(deepcopy(profiles_manager))

    @staticmethod
    def _calculate_additional_trees(results: Med3paRecord,
                                    dataset_mdr: MaskedDataset,
                                    features: List[str],
                                    APC_model: APCModel,
                                    confidence_model: IPCModel | APCModel | MPCModel,
                                    samples_ratio_min: int,
                                    samples_ratio_max: int,
                                    samples_ratio_step: int,
                                    metrics_list: List[str],
                                    explainer: Explainer,
                                    random_state: Optional[int]):
        """
        Calculates the additional trees of profiles, with the same decision tree parameters as the APC model, and
        without explanations:

        - 'source_apc': tree of the source error model's confidence (the IPC trained on the reference dataset by the
          explainer), with its profiles on the reference dataset, compared with the same profiles in dataset_mdr:
          the values are 'reference (dataset_mdr)', with the mean confidence level in dataset_mdr given by the
          confidence model (the one of the APC tree);
        - 'covariate_shift' and 'outcome_shift': trees of the per-patient covariate and outcome shifts of the
          explainer's shift test (see ShiftExplainer.get_shift_values), with their profiles on dataset_mdr (and the
          metrics of the same profiles in the reference dataset).

        Each tree is only calculated when the explainer provides its values (e.g. a ShiftExplainer with the IPC model
        as error model).

        Args:
            results (Med3paRecord): The results record.
            dataset_mdr (MaskedDataset): The MDR dataset.
            features (List[str]): The features of the dataset.
            APC_model (APCModel): The APC model, whose decision tree parameters are reused.
            confidence_model (IPCModel | APCModel | MPCModel): The confidence model, filtering the profiles of the
                shift trees by declaration rate.
            samples_ratio_min (int): Minimum sample ratio.
            samples_ratio_max (int): Maximum sample ratio.
            samples_ratio_step (int): Step size for sample ratio.
            metrics_list (List[str]): List of metrics to calculate.
            explainer (Explainer): The explainer of the experiment.
            random_state (int, optional): The random state of the decision trees.
        """
        reference_dataset = explainer.reference_dataset
        # name: (dataset of the profiles, values fitted by the tree, confidence scores, explainer of the metrics,
        # label of the mean value in the node information, compare the mean confidence level with dataset_mdr)
        trees = {}
        source_error_model = explainer.get_source_error_model()
        shift_values = explainer.get_shift_values(dataset_mdr.get_observations())
        if (source_error_model is None or reference_dataset is None) and shift_values is None:
            return
        confidence_preds = confidence_model.predict(dataset_mdr.get_observations())

        if source_error_model is not None and reference_dataset is not None:
            source_confidence = source_error_model.predict(reference_dataset.get_observations())
            # no explanation method, only the metrics of the same profiles in dataset_mdr
            trees['source_apc'] = (reference_dataset, source_confidence, source_confidence,
                                   Explainer(reference_dataset=dataset_mdr, methods=[]), None, True)

        if shift_values is not None:
            # no explanation method, only the metrics of the same profiles in the reference dataset
            metrics_explainer = Explainer(reference_dataset=reference_dataset, methods=[])
            for name, values in shift_values.items():
                if values is not None:
                    trees[name] = (dataset_mdr, values, confidence_preds, metrics_explainer,
                                   f"Mean {name.replace('_', ' ')}", False)

        for name, (dataset, values, confidence_scores, metrics_explainer, value_label,
                   compare_confidence) in trees.items():
            tree_model = APCModel(features=features, params=deepcopy(APC_model.params), random_state=random_state)
            tree_model.train(dataset.get_observations(), values)
            profiles_manager = ProfilesManager(features)
            for samples_ratio in range(samples_ratio_min, samples_ratio_max + 1, samples_ratio_step):
                MDRCalculator.calc_profiles(profiles_manager=profiles_manager, tree=tree_model.treeRepresentation,
                                            dataset=dataset, features=features, confidence_scores=confidence_scores,
                                            min_samples_ratio=samples_ratio, metrics_list=metrics_list,
                                            explainer=metrics_explainer)
            if value_label is not None:
                Med3paExperiment._add_profiles_mean_value(profiles_manager=profiles_manager, dataset=dataset,
                                                          values=values, value_label=value_label)
            if compare_confidence:
                Med3paExperiment._add_profiles_compared_confidence(profiles_manager=profiles_manager,
                                                                   comparison_dataset=dataset_mdr,
                                                                   comparison_confidence=confidence_preds)
            results.set_additional_tree(name=name, tree=tree_model.treeRepresentation,
                                        profiles_manager=deepcopy(profiles_manager))
            print(f"Additional tree of profiles calculated: {name}")

    @staticmethod
    def _add_profiles_mean_value(profiles_manager: ProfilesManager, dataset: MaskedDataset, values: np.ndarray,
                                 value_label: str):
        """
        Adds the mean of `values` over all the samples of each profile (whatever the declaration rate) to its node
        information, under `value_label`.
        """
        mean_values = {}
        for profiles_by_dr in profiles_manager.get_profiles().values():
            for profiles in profiles_by_dr.values():
                for profile in profiles:
                    path = tuple(profile.path)
                    if path not in mean_values:
                        mask = dataset.mask_by_profile(path=profile.path)
                        mean_values[path] = float(np.mean(values[mask])) if np.any(mask) else None
                    if profile.node_information is not None:
                        profile.node_information[value_label] = mean_values[path]

    @staticmethod
    def _add_profiles_compared_confidence(profiles_manager: ProfilesManager, comparison_dataset: MaskedDataset,
                                          comparison_confidence: np.ndarray):
        """
        Replaces the mean confidence level of each profile by 'x (y)', with y the mean of `comparison_confidence`
        (in %) over all the samples of the same profile in `comparison_dataset`, as the metrics compared with the
        reference dataset in MDRCalculator.
        """
        def fmt(x):
            return "" if x is None else f"{x:.2f}"

        mean_confidences = {}
        for profiles_by_dr in profiles_manager.get_profiles().values():
            for profiles in profiles_by_dr.values():
                for profile in profiles:
                    info = profile.node_information
                    # the same profile can be in several declaration rates: formatted once
                    if info is None or isinstance(info.get('Mean confidence level'), str):
                        continue
                    path = tuple(profile.path)
                    if path not in mean_confidences:
                        mask = comparison_dataset.mask_by_profile(path=profile.path)
                        mean_confidences[path] = float(np.mean(comparison_confidence[mask])) * 100 if np.any(
                            mask) else None
                    info['Mean confidence level'] = (f"{fmt(info.get('Mean confidence level'))} "
                                                     f"({fmt(mean_confidences[path])})")

    @staticmethod
    def _evaluate_models(results: Med3paRecord,
                         IPC_model: IPCModel,
                         APC_model: APCModel,
                         MPC_model: MPCModel,
                         dataset_evaluate: Optional[MaskedDataset],
                         confidence_calc: ConfidenceCalculator,
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
            confidence_calc (ConfidenceCalculator): The confidence calculator.
            evaluate_models (bool): Whether to evaluate the models.
            models_metrics (List[str]): List of metrics for model evaluation.
        """
        if evaluate_models:
            if not isinstance(dataset_evaluate, MaskedDataset):
                raise ValueError(f"Wrong data input to evaluate confidence models: {type(dataset_evaluate)}")
            confidence_evaluate = confidence_calc.calculate_confidence(dataset_evaluate.get_observations(),
                                                                          dataset_evaluate.get_pseudo_probabilities(),
                                                                          dataset_evaluate.get_true_labels())
            IPC_evaluation = IPC_model.evaluate(X=dataset_evaluate.get_observations(), y=confidence_evaluate,
                                                eval_metrics=models_metrics)
            APC_evaluation = APC_model.evaluate(X=dataset_evaluate.get_observations(), y=confidence_evaluate,
                                                eval_metrics=models_metrics)
            MPC_evaluation = MPC_model.evaluate(X=dataset_evaluate.get_observations(), y=confidence_evaluate,
                                                eval_metrics=models_metrics)
            results.set_models_evaluation(IPC_evaluation, APC_evaluation, MPC_evaluation)