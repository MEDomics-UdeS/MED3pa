Installation Guide
==========================================

Welcome to the installation guide of ``MED3pa`` package. Follow the steps below to install and get started with the package.

Prerequisites
-------------

Before installing the package, ensure you have the following prerequisites:

- Python 3.9 or later
- pip (Python package installer)

Installation
------------

Step 1: Install via pip
~~~~~~~~~~~~~~~~~~~~~~~

You can install ``MED3pa`` directly from PyPI using pip::

    pip install med3pa

Step 2: Verify the Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To verify that the installation was successful, you can run the following command::

    python -c "import MED3pa; print('Installation successful!')"

If you see `Installation successful!`, then you are ready to start using ``MED3pa``.

Usage
-----

Basic Example
~~~~~~~~~~~~~

Here is a basic example to get you started::

    from MED3pa.datasets import DatasetsManager
    from MED3pa.models import BaseModelManager, ModelFactory
    from MED3pa.med3pa import Med3paDetectronExperiment
    
    # Initialize the DatasetsManager
    datasets = DatasetsManager()

    # Load datasets for training, validation, reference, and testing
    datasets.set_from_file(dataset_type="training", file='./tutorials/data/train_data.csv', target_column_name='Outcome')
    datasets.set_from_file(dataset_type="validation", file='./tutorials/data/val_data.csv', target_column_name='Outcome')
    datasets.set_from_file(dataset_type="reference", file='./tutorials/data/test_data.csv', target_column_name='Outcome')
    datasets.set_from_file(dataset_type="testing", file='./tutorials/data/test_data_shifted_0.6.csv', target_column_name='Outcome')

    # Initialize the model factory and load the pre-trained model
    factory = ModelFactory()
    model = factory.create_model_from_pickled("./tutorials/models/diabetes_xgb_model.pkl")

    # Set the base model using BaseModelManager
    base_model_manager = BaseModelManager()
    base_model_manager.set_base_model(model=model)

    # Execute the integrated MED3PA and Detectron experiment
    reference_results, test_results, detectron_results = Med3paDetectronExperiment.run(
        datasets=datasets,
        base_model_manager=base_model_manager,
    )

    # Save the results to a specified directory
    reference_results.save(file_path='./tutorials/med3pa_detectron_experiment_results/reference')
    test_results.save(file_path='./tutorials/med3pa_detectron_experiment_results/test')
    detectron_results.save(file_path='./tutorials/med3pa_detectron_experiment_results/detectron')
