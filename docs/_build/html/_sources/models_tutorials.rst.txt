Working with the Models Subpackage
----------------------------------

The ``models`` subpackage is crafted to offer a comprehensive suite of tools for creating and managing various machine learning models within the ``MED3pa`` package.

Using the ModelFactory Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``ModelFactory`` class within the ``models`` subpackage offers a streamlined approach to creating machine learning models, either from predefined configurations or from serialized states. Here’s how to leverage this functionality effectively:

Step 1: Importing Necessary Modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Start by importing the required classes and utilities for model management:

.. code-block:: python

    from pprint import pprint
    from MED3pa.models import factories

Step 2: Creating an Instance of ModelFactory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Instantiate the ``ModelFactory``, which serves as your gateway to generating various model instances:

.. code-block:: python

    factory = factories.ModelFactory()

Step 3: Discovering Supported Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Before creating a model, check which models are currently supported by the factory:

.. code-block:: python

    print("Supported models:", factory.get_supported_models())

**Output**:

.. code-block:: none

    Supported models: ['XGBoostModel']

With this knowledge, we can proceed to create a model with specific hyperparameters.

Step 4: Specifying and Creating a Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Define hyperparameters for an XGBoost model and use these to instantiate a model:

.. code-block:: python

    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'nthread': 4,
        'tree_method': 'hist',
        'device': 'cpu'
    }

    xgb_model = factory.create_model_with_hyperparams('XGBoostModel', xgb_params)

Now, let’s inspect the model's configuration:

.. code-block:: python

    pprint(xgb_model.get_info())

**Output**:

.. code-block:: none

    {'data_preparation_strategy': 'ToDmatrixStrategy',
     'model': 'XGBoostModel',
     'model_type': 'Booster',
     'params': {'colsample_bytree': 0.8,
            'device': 'cpu',
            'eta': 0.1,
            'eval_metric': 'auc',
            'max_depth': 6,
            'min_child_weight': 1,
            'nthread': 4,
            'objective': 'binary:logistic',
            'subsample': 0.8,
            'tree_method': 'hist'},
     'pickled_model': False}

This gives us general information about the model, such as its ``data_preparation_strategy``, indicating that the input data for training, prediction, and evaluation will be transformed to ``Dmatrix`` to better suit the ``xgb.Booster`` model. It also retrieves the model's parameters, the underlying model instance class (``Booster`` in this case), and the wrapper class (``XGBoostModel`` in this case). Finally, it indicates whether this model has been created from a pickled file.

Step 5: Loading a Model from a Serialized State
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For pre-trained models, we can make use of the ``create_model_from_pickled`` method to load a model from its serialized (pickled) state. You only need to specify the path to this pickled file. This function will examine the pickled file and extract all necessary information.

.. code-block:: python

    xgb_model_pkl = factory.create_model_from_pickled('path_to_model.pkl')
    pprint(xgb_model_pkl.get_info())

**Output**:

.. code-block:: none

    {'data_preparation_strategy': 'ToDmatrixStrategy',
     'model': 'XGBoostModel',
     'model_type': 'Booster',
     'params': {'alpha': 0,
            'base_score': 0.5,
            'boost_from_average': 1,
            'booster': 'gbtree',
            'cache_opt': 1,
            ...
            'updater': 'grow_quantile_histmaker',
            'updater_seq': 'grow_quantile_histmaker',
            'validate_parameters': 0},
     'pickled_model': True}

Using the Model Class
~~~~~~~~~~~~~~~~~~~~~
In this section, we will learn how to train, predict, and evaluate a machine learning model. For this, we will directly use the created model from the previous section.

Step 1: Training the Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Generate Training and Validation Data:

Prepare the data for training and validation. The following example generates synthetic data for demonstration purposes:

.. code-block:: python

    np.random.seed(0)
    X_train = np.random.randn(1000, 10)
    y_train = np.random.randint(0, 2, 1000)
    X_val = np.random.randn(1000, 10)
    y_val = np.random.randint(0, 2, 1000)

Training the Model:

When training a model, you can specify additional ``training_parameters``. If they are not specified, the model will use the initialization parameters. You can also specify whether you'd like to balance the training classes.

.. code-block:: python

    training_params = {
        'eval_metric': 'logloss',
        'eta': 0.1,
        'max_depth': 6
    }
    xgb_model.train(X_train, y_train, X_val, y_val, training_params, balance_train_classes=True)

This process optimizes the model based on the specified hyperparameters and validation data to prevent overfitting.

Step 2: Predicting Using the Trained Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Model Prediction:

Once the model is trained, use it to predict labels or probabilities on a new dataset. This step demonstrates predicting binary labels for the test data. The ``return_proba`` parameter specifies whether to return the ``predicted_probabilities`` or the ``predicted_labels``. The labels are calculated based on the ``threshold``.

.. code-block:: python

    X_test = np.random.randn(1000, 10)
    y_test = np.random.randint(0, 2, 1000)
    y_pred = xgb_model.predict(X_test, return_proba=False, threshold=0.5)

Step 3: Evaluating the Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Evaluate the model's performance using various metrics to understand its effectiveness in making predictions. The supported metrics include Accuracy, AUC, Precision, Recall, and F1 Score, among others. The ``evaluate`` method will handle the model predictions and then evaluate the model based on these predictions. You only need to specify the test data.

To retrieve the list of supported ``classification_metrics``, you can use ``ClassificationEvaluationMetrics.supported_metrics()``:

.. code-block:: python

    from MED3pa.models import ClassificationEvaluationMetrics

    # Display supported metrics
    print("Supported evaluation metrics:", ClassificationEvaluationMetrics.supported_metrics())

    # Evaluate the model
    evaluation_results = xgb_model.evaluate(X_test, y_test, eval_metrics=['Auc', 'Accuracy'], print_results=True)

**Output**:

.. code-block:: none

    Supported evaluation metrics: ['Accuracy', 'BalancedAccuracy', 'Precision', 'Recall', 'F1Score', 'Specificity', 'Sensitivity', 'Auc', 'LogLoss', 'Auprc', 'NPV', 'PPV', 'MCC']
    Evaluation Results:
    Auc: 0.51
    Accuracy: 0.50

Step 4: Retrieving Model Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``get_info`` method provides detailed information about the model, including its type, parameters, data preparation strategy, and whether it's a pickled model. This is useful for understanding the configuration and state of the model.

.. code-block:: python

    model_info = xgb_model.get_info()
    pprint(model_info)

**Output**:

.. code-block:: none

    {'model': 'XGBoostModel',
     'model_type': 'Booster',
     'params': {'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'eta': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'nthread': 4,
                'tree_method': 'hist',
                'device': 'cpu'},
     'data_preparation_strategy': 'ToDmatrixStrategy',
     'pickled_model': False}

Step 5: Saving Model Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can save the model by using the `save` method, which will save the underlying model instance as a pickled file, and the model's information as a .json file:

.. code-block:: none

    xgb_model.save("./models/saved_model")