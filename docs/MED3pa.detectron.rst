detectron subpackage
========================
Overview
---------
The ``detectron`` subpackage is a modular and enhanced implementation of the Detectron method, as described in the paper  `"Ginsberg, T., Liang, Z., & Krishnan, R. G. (2023). A Learning Based Hypothesis Test for Harmful Covariate Shift." <https://openreview.net/forum?id=rdfgqiwz7lZ>`__. 
This method is designed to detect potentially harmful shifts in data distributions that could undermine the reliability and performance of machine learning models in critical applications.

**Detectron** employs a set of classifiers called (CDCs) trained to agree on domain data but explicitly designed to disagree on this possibly shifted data. 
This unique strategy allows it to effectively identify and quantify shifts in data distributions that pose a risk to the model's generalization capabilities. 

The ``detectron`` subpackage offers a fully-featured suite for seamlessly incorporating the Detectron method into existing machine learning pipelines. 
It enables the robust and automated detection of harmful covariate shifts through an array of functional modules. 

These include sophisticated tools for managing ensemble classifiers, meticulously recording results, and applying diverse statistical strategies for shift evaluation:

- **ensemble.py**: Manages the ensemble of Constrained Disagreement Classifiers (CDCs) that challenge the predictions of a primary base model.

- **record.py**: Handles the storage and management of Detectron results across different runs, tracking metrics and probabilities.

- **stopper.py**: Provides a utility for early stopping to prevent overfitting by halting training when improvements cease.

- **strategies.py**: Defines various strategies for evaluating the presence of covariate shifts between calibration and testing datasets using statistical tests.

- **experiment.py**: Orchestrates the setup and execution of Detectron method, managing the flow of data, model training, and evaluations.

- **comparaison.py**: Compares or aggregates the results of two DetectronExperiments.


this subpackage includes the following classes:

.. image:: ./diagrams/detectron.svg
   :alt: UML class diagram of the subpackage.
   :align: center

.. raw:: html

   <div style="margin-bottom: 30px;"></div>

ensemble module
--------------------------------

.. automodule:: MED3pa.detectron.ensemble
   :members:
   :undoc-members:
   :show-inheritance:

record module
------------------------------

.. automodule:: MED3pa.detectron.record
   :members:
   :undoc-members:
   :show-inheritance:

stopper module
-------------------------------

.. automodule:: MED3pa.detectron.stopper
   :members:
   :undoc-members:
   :show-inheritance:

strategies module
----------------------------------

.. automodule:: MED3pa.detectron.strategies
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

experiment module
----------------------------------

.. automodule:: MED3pa.detectron.experiment
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

comparaison module
----------------------------------

.. automodule:: MED3pa.detectron.comparaison
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex: