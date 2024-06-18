Working with the Detectron Subpackage
=====================================
This tutorial guides you through the process of setting up and running the Detectron experiment using the ``detectron`` subpackage, which is designed to assess the robustness of predictive models against covariate shifts in datasets.

Step 1: Setting up the Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
First, configure the ``DatasetsManager`` to manage the necessary datasets for the experiment. For this, we need the ``training_dataset`` used to train the model, alongside the ``validation_dataset``. We also need an unseen ``reference_dataset`` from the model's domain, and finally the dataset we want to inspect if it is possibly shifted, the ``test_dataset``.

.. code-block:: python

    from det3pa.datasets import DatasetsManager

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

    from det3pa.models import BaseModelManager, ModelFactory

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

    from det3pa.detectron import DetectronExperiment, DisagreementStrategy_z_mean

    # Execute the Detectron experiment
    experiment_results = DetectronExperiment.run(datasets=datasets, base_model_manager=base_model_manager)

Step 4: Analyzing the Results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Finally, evaluate the outcomes of the experiment using a specified strategy to determine the probability of a shift in dataset distributions.

.. code-block:: python

    # Analyze the results using the chosen disagreement strategy
    experiment_results.analyze_results(DisagreementStrategy_z_mean)

**Output**

The following output provides a detailed assessment of dataset stability:

.. code-block:: none

    "shift_probability": 0.93,
    "test_statistic": 11.15,
    "baseline_mean": 7.78,
    "baseline_std": 2.555699512853575,
    "significance_description": {
        "no shift": 14.760000000000002,
        "small": 25.39,
        "moderate": 37.93,
        "large": 21.92
    },
    "Strategy": "DisagreementStrategy_z_mean"}

