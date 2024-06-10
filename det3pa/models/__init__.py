"""
The `models` sub-package provides various classes and functions for managing, training, and evaluating machine learning models. It encompasses a range of functionalities from base model definitions to specific classifiers and regressors, as well as utilities for measuring performance.

Modules:
    abstract_models: Contains abstract base classes for model definitions, ensuring a consistent interface across different types of models.
    base: Includes foundational classes and functions that are extended by more specific model implementations.
    concrete_classifiers: Provides implementations of specific classification models tailored to various machine learning tasks.
    concrete_regressors: Offers implementations of regression models for continuous data prediction.
    classification_metrics: Contains functions and classes to evaluate the performance of classification models using various metrics.
    regression_metrics: Includes methods to assess regression model performance through standard and custom metrics.
    factories: Facilitates the creation of models and related objects dynamically, supporting flexible and configurable model instantiation.
"""

from .abstract_models import *
from .base import *
from .concrete_classifiers import *
from .concrete_regressors import *
from .classification_metrics import *
from .regression_metrics import *
from .factories import *
