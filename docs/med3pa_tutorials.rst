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
    reference_results, test_results = Med3paExperiment.run(
                                        datasets_manager=datasets,
                                        base_model_manager=base_model_manager,
                                        uncertainty_metric=AbsoluteError,
                                        ipc_type=RandomForestRegressorModel,
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
    reference_results.save(file_path='./med3pa_experiment_results/reference')
    test_results.save(file_path='./med3pa_experiment_results/test')


Running the MED3pa and Detectron Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
You can also run an experiment that combines the forces of Detectron in covariate shift detection with MED3pa problematic profiles extraction using the `Med3paDetectronExperiment` class. To be able to run this experiment, all datasets of the `DatasetsManager` should be set, alongside the ``BaseModelManager``. This experiment will run MED3pa experiment on the `testing` and `reference` sets and then run the `detectron` experiment on the `testing` set as a whole, and then on the **extracted profiles** from MED3pa:

.. code-block:: python

    from MED3pa.med3pa import Med3paDetectronExperiment
    from MED3pa.detectron.strategies import EnhancedDisagreementStrategy

    # Execute the integrated MED3PA and Detectron experiment
    reference_results, test_results, detectron_results = Med3paDetectronExperiment.run(
        datasets=datasets,
        base_model_manager=base_model_manager,
        uncertainty_metric=AbsoluteError,
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
    reference_results.save(file_path='./med3pa_detectron_experiment_results/reference')
    test_results.save(file_path='./med3pa_detectron_experiment_results/test')
    detectron_results.save(file_path='./med3pa_detectron_experiment_results/detectron')
