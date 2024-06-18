Working with the med3pa Subpackage
----------------------------------
This tutorial guides you through the process of setting up and running comprehensive experiments using the ``med3pa`` subpackage. It includes steps to execute MED3pa experiment with ``Med3paExperiment`` and  the combination of MED3pa and Detectron using ``Med3paDetectronExperiment``.

Running the MED3pa Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Step 1: Setting up the Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
First, configure the ``DatasetsManager``. In the case of MED3pa experiment you only need to set the DatasetManager with either ``testing`` or ``reference`` dataset. 

.. code-block:: python

    from det3pa.datasets import DatasetsManager

    # Initialize the DatasetsManager
    datasets = DatasetsManager()

    # Load dataset to use in the experiment
    datasets.set_from_file(dataset_type="testing", file='./path_to_test_dataset.csv', target_column_name='y_true')

Step 2: Configuring the Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Next, utilize the ``ModelFactory`` to load a pre-trained model, and set it as the base model for the experiment. Alternatively, you can train your own model and use it.

.. code-block:: python

    from det3pa.models import BaseModelManager, ModelFactory

    # Initialize the model factory and load the pre-trained model
    factory = ModelFactory()
    model = factory.create_model_from_pickled("./path_to_model.pkl")

    # Set the base model using BaseModelManager
    base_model_manager = BaseModelManager()
    base_model_manager.set_base_model(model=model)

Step 3: Running the med3pa Experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Execute the MED3PA experiment with the specified datasets and base model. You can also specify other parameters as needed. see the documention of the subpackage for more information about the parameters

.. code-block:: python

    from det3pa.med3pa import Med3paExperiment
    from det3pa.med3pa.uncertainty import AbsoluteError
    from det3pa.models.concrete_regressors import RandomForestRegressorModel

    # Define parameters for the experiment
    ipc_params = {'n_estimators': 100}
    apc_params = {'max_depth': 3}
    med3pa_metrics = ['Auc', 'Accuracy', 'BalancedAccuracy']

    # Execute the MED3PA experiment
    experiment_results = Med3paExperiment.run(
        datasets_manager=datasets,
        set='testing',
        base_model_manager=base_model_manager,
        uncertainty_metric=AbsoluteError,
        ipc_type=RandomForestRegressorModel,
        ipc_params=ipc_params,
        apc_params=apc_params,
        samples_ratio_min=0,
        samples_ratio_max=50,
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
    experiment_results.save(file_path='./med3pa_experiment_results')


Running the MED3pa and Detectron Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also run an experiment that combines the forces of Detectron in covariate shift detection with MED3pa problematic profiles extraction using the ``Med3paDetectronExperiment`` class.
To be able to run this experiment, all datasets of the ``DatasetsManager`` should be set. 
This experiment will run MED3pa experiment on the ``testing`` and ``reference`` sets and then run the ``detectron`` experiment on the ``testing`` set, and finally run ``detectron`` on the **extracted profiles** from MED3pa.

.. code-block:: python

    from det3pa.detectron import DisagreementStrategy_z_mean

    # Define additional parameters for the Detectron experiment
    training_params = {'eval_metric': 'logloss', 'eta': 0.1, 'max_depth': 6}

    # Execute the integrated MED3PA and Detectron experiment
    reference_3pa_res, testing_3pa_res, detectron_results = Med3paDetectronExperiment.run(
        datasets=datasets,
        training_params=training_params,
        base_model_manager=base_model_manager,
        uncertainty_metric=AbsoluteError,
        samples_size=20,
        ensemble_size=10,
        num_calibration_runs=100,
        patience=3,
        test_strategy=DisagreementStrategy_z_mean,
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