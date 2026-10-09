"""
Error models shared by the explanation methods: training a new instance of a MED3pa confidence model (e.g. the IPC
model of an experiment) on another dataset, e.g. the source error model trained on the reference dataset.
"""
import numpy as np
from copy import deepcopy
from typing import Any


def fit_confidence_model(error_model: Any, f_model: Any, x: np.ndarray, y: np.ndarray,
                         confidence: np.ndarray) -> Any:
    """
    New instance of the class of `error_model` (e.g. an IPCModel), with the same regressor and parameters, trained
    to predict the `confidence` of the predictions of `f_model` on `x`, with the same sample weights as in
    ``Med3paExperiment._train_models``.

    Args:
        error_model (Any): The trained confidence model whose class, regressor and parameters are reused.
        f_model (Any): The model whose predictions' confidence is predicted, with a ``predict_proba`` method.
        x (np.ndarray): Training observations.
        y (np.ndarray): True labels of the training observations.
        confidence (np.ndarray): Confidence scores of the predictions of `f_model` on `x` (the targets).

    Returns:
        Any: The new trained confidence model.
    """
    model = type(error_model)(model_name=error_model.model_name, params=deepcopy(error_model.params))
    train_params = {}
    if error_model.model_name == 'EnsembleRandomForestRegressor':
        class_1_prop = int(np.sum(y)) / len(y)
        sample_weight = np.where(y == 0, 1 / (1 - class_1_prop), 1 / class_1_prop)
    else:
        sample_weight = None
        if error_model.model_name == 'BalancedEnsembleRandomForestRegressor':
            # class and error weights computed from the true labels and the labels predicted by f_model
            threshold = getattr(f_model, "threshold", 0.5)
            train_params = {"y_true": y, "y_pred": (f_model.predict_proba(x)[:, 1] >= threshold).astype(int)}
    model.train(x, confidence, sample_weight=sample_weight, **train_params)
    return model
