Working with datasets subpackage
--------------------------------
The ``datasets`` subpackage is designed to provide robust and flexible data loading and management functionalities tailored for machine learning models. 
This tutorial will guide you through using this subpackage to handle and prepare your data efficiently.

Using the DatasetsManager class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``DatasetsManager`` class in the ``MED3pa.datasets`` submodule is designed to facilitate the management of various datasets needed for model training and evaluation. This tutorial provides a step-by-step guide on setting up and using the ``DatasetsManager`` to handle data efficiently.

Step 1: Importing the DatasetsManager
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, import the ``DatasetsManager`` from the ``MED3pa.datasets`` submodule:

.. code-block:: python

    from MED3pa.datasets import DatasetsManager

Step 2: Creating an Instance of DatasetsManager
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Create an instance of ``DatasetsManager``. This instance will manage all operations related to datasets:

.. code-block:: python

    manager = DatasetsManager()

Step 3: Loading Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
With the ``DatasetsManager``, you can load various segments of your base model datasets, such as training, validation, reference, and testing datasets. You don't need to load all datasets at once. Provide the path to your dataset and the name of the target column:

**Loading from File**

.. code-block:: python

    manager.set_from_file(dataset_type="training", file='./path_to_training_dataset.csv', target_column_name='target_column')

**Loading from NumPy Arrays**

You can also load the datasets as NumPy arrays. For this, you need to specify the features, true labels, and column labels as a list (excluding the target column) if they are not already set.

.. code-block:: python

    import numpy as np
    import pandas as pd

    df = pd.read_csv('./path_to_validation_dataset.csv')

    # Extract labels and features
    X_val = df.drop(columns='target_column').values
    y_val = df['target_column'].values

    # Example of setting data from numpy arrays
    manager.set_from_data(dataset_type="validation", observations=X_val, true_labels=y_val)

Step 4: Ensuring Feature Consistency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Upon loading the first dataset, the ``DatasetsManager`` automatically extracts and stores the names of features. You can retrieve the list of these features using:

.. code-block:: python

    features = manager.get_column_labels()

Ensure that the features of subsequent datasets (e.g., validation or testing) match those of the initially loaded dataset to avoid errors and maintain data consistency.

Step 5: Retrieving Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Retrieve the loaded data in different formats as needed.

**As NumPy Arrays**

.. code-block:: python

    observations, labels = manager.get_dataset_by_type(dataset_type="training")

**As a MaskedDataset Instance**

To work with the data encapsulated in a ``MaskedDataset`` instance, which might include more functionalities, retrieve it by setting ``return_instance`` to ``True``:

.. code-block:: python

    training_dataset = manager.get_dataset_by_type(dataset_type="training", return_instance=True)

Step 6: Getting a Summary
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can print a summary of the ``DatasetsManager`` to see the status of the datasets:

.. code-block:: python

    manager.summarize()

Step 7: Saving and Resetting Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can save a specific dataset to a CSV file or reset all datasets managed by the ``DatasetsManager``.

**Save to CSV**

.. code-block:: python

    manager.save_dataset_to_csv(dataset_type="training", file_path='./path_to_save_training_dataset.csv')

**Reset Datasets**

.. code-block:: python

    manager.reset_datasets()
    manager.summarize()  # Verify that all datasets are reset

Summary of Outputs
^^^^^^^^^^^^^^^^^^^

When you run the ``summary`` method, you should get an output similar to this, indicating the status and details of each dataset:

.. code-block:: none

    training_set: {'num_samples': 151, 'num_features': 23, 'has_pseudo_labels': False, 'has_pseudo_probabilities': False, 'has_confidence_scores': False}
    validation_set: {'num_samples': 1000, 'num_features': 10, 'has_pseudo_labels': False, 'has_pseudo_probabilities': False, 'has_confidence_scores': False}
    reference_set: Not set
    testing_set: Not set
    column_labels: ['feature_1', 'feature_2', ..., 'feature_23']

Using the MaskedDataset Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``MaskedDataset`` class, a crucial component of the ``MED3pa.datasets`` submodule, facilitates nuanced data operations that are essential for custom data manipulation and model training processes. This tutorial details common usage scenarios of the ``MaskedDataset``.

Step 1: Importing Necessary Modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Begin by importing the ``MaskedDataset`` and ``DatasetsManager``, along with NumPy for additional data operations:

.. code-block:: python

    from MED3pa.datasets import MaskedDataset, DatasetsManager
    import numpy as np

Step 2: Loading Data with DatasetsManager
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Retrieve the dataset as a ``MaskedDataset`` instance:

.. code-block:: python

    manager = DatasetsManager()
    manager.set_from_file(dataset_type="training", file='./path_to_training_dataset.csv', target_column_name='target_column')
    training_dataset = manager.get_dataset_by_type(dataset_type="training", return_instance=True)

Step 3: Performing Operations on MaskedDataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Once you have your dataset loaded as a ``MaskedDataset`` instance, you can perform various operations:

**Cloning the Dataset**

Create a copy of the dataset to ensure the original data remains unchanged during experimentation:

.. code-block:: python

    cloned_instance = training_dataset.clone()

**Sampling the Dataset**

Randomly sample a subset of the dataset, useful for creating training or validation splits:

.. code-block:: python

    sampled_instance = training_dataset.sample(N=20, seed=42)

**Refining the Dataset**

Refine the dataset based on a boolean mask, which is useful for filtering out unwanted data points:

.. code-block:: python

    mask = np.random.rand(len(training_dataset)) > 0.5
    remaining_samples = training_dataset.refine(mask=mask)

**Setting Pseudo Labels and Probabilities**

Set pseudo labels and probabilities for the dataset, for this you only need to pass the pseudo_probabilities along with the threshold to extract the pseudo_labels from:

.. code-block:: python

    pseudo_probs = np.random.rand(len(training_dataset))
    training_dataset.set_pseudo_probs_labels(pseudo_probabilities=pseudo_probs, threshold=0.5)

**Getting Feature Vectors and Labels**

Retrieve the feature vectors, true labels, and pseudo labels:

.. code-block:: python

    observations = training_dataset.get_observations()
    true_labels = training_dataset.get_true_labels()
    pseudo_labels = training_dataset.get_pseudo_labels()

**Getting Confidence Scores**

Get the confidence scores if available:

.. code-block:: python

    confidence_scores = training_dataset.get_confidence_scores()

**Converting to DataFrame and Saving to CSV**

### Saving the dataset
You can save the dataset as a .csv file, but using `save_to_csv` and providing the path this will save the observations, true_labels, pseudo_labels and pseudo_probabilities, alongside confidence_scores if they were set:

.. code-block:: python

    df = training_dataset.to_dataframe()
    training_dataset.save_to_csv('./path_to_save_training_dataset.csv')

**Getting Dataset Information**

Get detailed information about the dataset, or you can directly use ``summary``:

.. code-block:: python

    training_dataset.summarize()

When you run the ``summarize`` method, you should get an output similar to this, indicating the status and details of the dataset:

.. code-block:: none
    
    Number of samples: 151
    Number of features: 23
    Has pseudo labels: False
    Has pseudo probabilities: False
    Has confidence scores: False
