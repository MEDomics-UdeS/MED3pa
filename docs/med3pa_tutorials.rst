Working with the med3pa Subpackage
----------------------------------
This tutorial guides you through the process of setting up and running comprehensive experiments using the ``med3pa`` subpackage. It includes steps to execute MED3pa experiment with ``Med3paExperiment`` and  the combination of MED3pa and Detectron using ``Med3paDetectronExperiment``.

Running the MED3pa Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Step 1: Setting up the Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
First, configure the `DatasetsManager`. In the case of MED3pa only experiment you only need to set the DatasetManager with either `testing` and `reference` dataset:

.. code-block:: python

    from MED3pa.datasets import DatasetsManager

    # Initialize the DatasetsManager
    datasets = DatasetsManager()

    # Load datasets for reference, and testing
    datasets.set_from_file(dataset_type="reference", file='./path_to_reference_data.csv', target_column_name='Outcome')
    datasets.set_from_file(dataset_type="testing", file='./path_to_test_data.6.csv', target_column_name='Outcome')

    # Initialize the DatasetsManager
    datasets2 = DatasetsManager()

    # Load datasets for reference, and testing
    datasets2.set_from_file(dataset_type="reference", file='./data/test_data.csv', target_column_name='Outcome')
    datasets2.set_from_file(dataset_type="testing", file='./data/test_data_shifted_1.6.csv', target_column_name='Outcome')

Step 2: Configuring the Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Next, utilize the ``ModelFactory`` to load a pre-trained model, and set it as the base model for the experiment. Alternatively, you can train your own model and use it.

.. code-block:: python

    from MED3pa.models import BaseModelManager, ModelFactory

    # Initialize the model factory and load the pre-trained model
    factory = ModelFactory()
    model = factory.create_model_from_pickled("./path_to_model.pkl")

    # Set the base model using BaseModelManager
    base_model_manager = BaseModelManager()
    base_model_manager.set_base_model(model=model)

Step 3: Running the med3pa Experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Execute the MED3PA experiment with the specified datasets and base model. You can also specify other parameters as needed. See the documentation of the subpackage for more information about the parameters.

The experiment outputs two structure one for the reference set and the other for the testing set, both containing files indicating the extracted profiles at different declaration rates, the performance of the model on these profiles..etc.

.. code-block:: python

    from MED3pa.med3pa import Med3paExperiment
    from MED3pa.med3pa.uncertainty import AbsoluteError
    from MED3pa.models.concrete_regressors import RandomForestRegressorModel

    # Define parameters for the experiment
    ipc_params = {'n_estimators': 100}
    apc_params = {'max_depth': 3}
    med3pa_metrics = ['Auc', 'Accuracy', 'BalancedAccuracy']

    # Execute the MED3PA experiment
    ipc_params = {'n_estimators': 100}
    apc_params = {'max_depth': 3}
    med3pa_metrics = ['Auc', 'Accuracy', 'BalancedAccuracy']

    # Execute the MED3PA experiment
    results = Med3paExperiment.run(
                                        datasets_manager=datasets,
                                        base_model_manager=base_model_manager,
                                        uncertainty_metric="absolute_error",
                                        ipc_type='RandomForestRegressor',
                                        ipc_params=ipc_params,
                                        apc_params=apc_params,
                                        samples_ratio_min=0,
                                        samples_ratio_max=10,
                                        samples_ratio_step=5,
                                        med3pa_metrics=med3pa_metrics,
                                        evaluate_models=True,
                                        models_metrics=['MSE', 'RMSE']
                                    )
    
    results2 = Med3paExperiment.run(
                                datasets_manager=datasets2,
                                base_model_manager=base_model_manager,
                                uncertainty_metric="absolute_error",
                                ipc_type='RandomForestRegressor',
                                ipc_params=ipc_params,
                                apc_params=apc_params,
                                samples_ratio_min=0,
                                samples_ratio_max=10,
                                samples_ratio_step=5,
                                med3pa_metrics=med3pa_metrics,
                                evaluate_models=True,
                                models_metrics=['MSE', 'RMSE']
                                )

Step 4: Analyzing and Saving the Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
After running the experiment, you can analyze and save the results using the returned ``Med3paResults`` instance.

.. code-block:: python

    # Save the results to a specified directory
    results.save(file_path='./med3pa_experiment_results/')
    results2.save(file_path='./med3pa_experiment_results_2')

Additonnally, you can save the instances the IPC and APC models as pickled files:

.. code-block:: python

    results.save_models(file_path='./med3pa_experiment_results_models')

Step 5: Running experiments from pretrained models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If you don't want to re-train new APC and IPC models in your experiment, you can directly use a previously saved instances. as follows:

.. code-block:: python

    from MED3pa.med3pa import Med3paExperiment
    from MED3pa.med3pa.uncertainty import AbsoluteError

    # Define parameters for the experiment
    ipc_params = {'n_estimators': 100}
    apc_params = {'max_depth': 3}
    med3pa_metrics = ['Auc', 'Accuracy', 'BalancedAccuracy']

    # Execute the MED3PA experiment
    results = Med3paExperiment.run(
                                    datasets_manager=datasets,
                                    base_model_manager=base_model_manager,
                                    uncertainty_metric="absolute_error",
                                    ipc_type='RandomForestRegressor',
                                    pretrained_ipc='./med3pa_experiment_results_models/ipc_model.pkl',
                                    pretrained_apc='./med3pa_experiment_results_models/apc_model.pkl',
                                    samples_ratio_min=0,
                                    samples_ratio_max=10,
                                    samples_ratio_step=5,
                                    med3pa_metrics=med3pa_metrics,
                                    evaluate_models=True,
                                    models_metrics=['MSE', 'RMSE']
                                    )

    results2 = Med3paExperiment.run(
                                    datasets_manager=datasets2,
                                    base_model_manager=base_model_manager,
                                    uncertainty_metric="absolute_error",
                                    ipc_type='RandomForestRegressor',
                                    pretrained_ipc='./med3pa_experiment_results_models/ipc_model.pkl',
                                    pretrained_apc='./med3pa_experiment_results_models/apc_model.pkl',
                                    samples_ratio_min=0,
                                    samples_ratio_max=10,
                                    samples_ratio_step=5,
                                    med3pa_metrics=med3pa_metrics,
                                    evaluate_models=True,
                                    models_metrics=['MSE', 'RMSE']
                                    )
    
    # Save the results to a specified directory
    results.save(file_path='./med3pa_experiment_results_pretrained')
    results2.save(file_path='./med3pa_experiment_results_2_pretrained')

Step 6: Comparing two experiments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can compare between two experiments bu using the ``Med3paComparaison`` class, this class works as follows:
- the two experiments need to be of the same type, either ``Med3paExperiment`` or ``Med3paDetectronExperiment``.
- if the two experiments were executed using the same tree structure, or the same apc/ipc models, the profiles will also be compared.
- if the experiments are of type ``Med3paDetectronExperiment``, the detectron results will be also compared.

.. code-block:: python

    from MED3pa.med3pa.comparaison import Med3paComparison

    comparaison = Med3paComparison('./med3pa_experiment_results_pretrained', './med3pa_experiment_results_2_pretrained')
    comparaison.compare_experiments()
    comparaison.save('./med3pa_comparaison_results')
    
Running the MED3pa and Detectron Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can also run an experiment that combines the forces of Detectron in covariate shift detection with MED3pa problematic profiles extraction using the `Med3paDetectronExperiment` class. To be able to run this experiment, all datasets of the `DatasetsManager` should be set, alongside the ``BaseModelManager``. This experiment will run MED3pa experiment on the `testing` and `reference` sets and then run the `detectron` experiment on the `testing` set as a whole, and then on the **extracted profiles** from MED3pa:

.. code-block:: python

    from MED3pa.med3pa import Med3paDetectronExperiment
    from MED3pa.detectron.strategies import EnhancedDisagreementStrategy

    # Execute the integrated MED3PA and Detectron experiment
    med3pa_results, detectron_results = Med3paDetectronExperiment.run(
        datasets=datasets,
        base_model_manager=base_model_manager,
        uncertainty_metric="absolute_error",
        samples_size=20,
        ensemble_size=10,
        num_calibration_runs=100,
        patience=3,
        test_strategies=EnhancedDisagreementStrategy,
        allow_margin=False,
        margin=0.05,
        ipc_params=ipc_params,
        apc_params=apc_params,
        samples_ratio_min=0,
        samples_ratio_max=50,
        samples_ratio_step=5,
        med3pa_metrics=med3pa_metrics,
        evaluate_models=True,
        models_metrics=['MSE', 'RMSE']
    )

    # Save the results to a specified directory
    med3pa_results.save(file_path='./med3pa_detectron_experiment_results/')
    detectron_results.save(file_path='./med3pa_detectron_experiment_results/detectron')
