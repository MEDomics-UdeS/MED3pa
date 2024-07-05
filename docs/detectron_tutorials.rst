Working with the Detectron Subpackage
=====================================
This tutorial guides you through the process of setting up and running the Detectron experiment using the ``detectron`` subpackage, which is designed to assess the robustness of predictive models against covariate shifts in datasets.

Step 1: Setting up the Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
First, configure the ``DatasetsManager`` to manage the necessary datasets for the experiment. For this, we need the ``training_dataset`` used to train the model, alongside the ``validation_dataset``. We also need an unseen ``reference_dataset`` from the model's domain, and finally the dataset we want to inspect if it is possibly shifted, the ``test_dataset``.

.. code-block:: python

    from MED3pa.datasets import DatasetsManager

    # Initialize the DatasetsManager
    datasets = DatasetsManager()

    # Load datasets for training, validation, reference, and testing
    datasets.set_from_file(dataset_type="training", file='./path_to_train_dataset.csv', target_column_name='y_true')
    datasets.set_from_file(dataset_type="validation", file='./path_to_validation_dataset.csv', target_column_name='y_true')
    datasets.set_from_file(dataset_type="reference", file='./path_to_reference_dataset.csv', target_column_name='y_true')
    datasets.set_from_file(dataset_type="testing", file='./path_to_test_dataset.csv', target_column_name='y_true')

Step 2: Configuring the Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Next, utilize the ``ModelFactory`` to load a pre-trained model, setting it as the base model for the experiment. Alternatively, you can train your own model and use it.

.. code-block:: python

    from MED3pa.models import BaseModelManager, ModelFactory

    # Initialize the model factory and load the pre-trained model
    factory = ModelFactory()
    model = factory.create_model_from_pickled("./path_to_model.pkl")

    # Set the base model using BaseModelManager
    base_model_manager = BaseModelManager()
    base_model_manager.set_base_model(model=model)

Step 3: Running the Detectron Experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Execute the Detectron experiment with the specified datasets and base model. You can also specify other parameters.

.. code-block:: python

    from MED3pa.detectron import DetectronExperiment

    # Execute the Detectron experiment
    experiment_results = DetectronExperiment.run(datasets=datasets, base_model_manager=base_model_manager)

Step 4: Analyzing the Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Finally, evaluate the outcomes of the experiment using different strategies to determine the probability of a shift in dataset distributions:

.. code-block:: python

    # Analyze the results using the disagreement strategies
    test_strategies = ["enhanced_disagreement_strategy", "mannwhitney_strategy"]
    experiment_results.analyze_results(test_strategies)

**Output**

The following output provides a detailed assessment of dataset stability:

.. code-block:: none

    [
        {
            "shift_probability": 0.8111111111111111,
            "test_statistic": 8.466666666666667,
            "baseline_mean": 7.4,
            "baseline_std": 1.2631530214330944,
            "significance_description": {
                "no shift": 38.34567901234568,
                "small": 15.592592592592592,
                "moderate": 16.34567901234568,
                "large": 29.716049382716047
            },
            "Strategy": "EnhancedDisagreementStrategy"
        },
        {
            "p_value": 0.00016360887668277182,
            "u_statistic": 3545.0,
            "z-score": 0.4685784328619402,
            "shift significance": "Small",
            "Strategy": "MannWhitneyStrategy"
        }
    ]

Step 5: Saving the Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can save the experiment results using the ``save`` method, while specifying the path.

.. code-block:: python
    
    experiment_results.save("./detectron_experiment_results")
