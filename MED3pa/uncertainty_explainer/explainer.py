"""

"""
import inspect
import numpy as np
from typing import Optional, List, Dict, Any

from MED3pa.datasets import MaskedDataset
from MED3pa.uncertainty_explainer.covariate_shit.detectron import Detectron
from MED3pa.uncertainty_explainer.shift.shift_explainer import ShiftExplainer
from MED3pa.uncertainty_explainer.uncertainty.ipc_uncertainty import IPCUncertainty
from MED3pa.uncertainty_explainer.abstract_methods import ExplanationMethod

class Explainer:
    """

    """
    __default_methods = [Detectron, ShiftExplainer, IPCUncertainty]
    def __init__(self, reference_dataset: Optional[MaskedDataset] = None,
                 f_model: Optional[Any] = None,
                 methods: Optional[List[ExplanationMethod]] = None):
        self._reference_dataset = reference_dataset
        self._methods = []
        if methods is None:
            methods = self.__default_methods

        for method in methods:
            if inspect.isclass(method):
                self._methods.append(method(reference_dataset=reference_dataset,
                                             f_model=f_model))
            else:
                if not method.context_is_set():
                    method.set_context(reference_dataset=reference_dataset, f_model=f_model)
                self._methods.append(method)

    def __call__(self, x_q:np.ndarray, profile_path:Optional[str] = None, y_q:Optional[np.ndarray] = None):
        """
        Applies all the explanation methods on x_q. The labels y_q are optional and only used by the methods
        that need them (e.g. the outcome-shift test of ShiftExplainer).
        """
        results = []

        for method in self._methods:
            result = method(x_q=x_q, profile_path=profile_path, y_q=y_q)
            results.append({
                'Method_name': method.get_name(),
                'Method_type': method.category,
                'result': result
            })
        return results

    def methods_count(self) -> int:
        """Number of explanation methods."""
        return len(self._methods)

    @property
    def reference_dataset(self):
        return self._reference_dataset

    def set_context(self, reference_dataset: MaskedDataset, f_model: Any):
        """

        """
        self._reference_dataset = reference_dataset
        for method in self._methods:
            method.set_context(reference_dataset=reference_dataset, f_model=f_model)

    def set_target_dataset(self, target_dataset: MaskedDataset):
        """
        Sets the whole target dataset (the dataset whose profiles are explained) for the methods that need it
        (e.g. ShiftExplainer, to fit its global models).
        """
        for method in self._methods:
            if hasattr(method, "set_target_dataset"):
                method.set_target_dataset(target_dataset=target_dataset)

    def set_error_model(self, error_model: Any, confidence_calc: Any, source_error_model: Optional[Any] = None):
        """
        Sets the trained error model (the IPC model of a MED3pa experiment) and the confidence calculator giving
        the confidence it predicts, for the methods that use them (e.g. ShiftExplainer, as its target error model).
        `source_error_model` is an already-trained source error model (e.g. from a previous run) to use instead of
        training one on the reference dataset.
        """
        for method in self._methods:
            if hasattr(method, "set_error_model"):
                method.set_error_model(error_model=error_model, confidence_calc=confidence_calc,
                                       source_error_model=source_error_model)

        # the methods using a source error model (e.g. IPCUncertainty) share the one of the first method training it
        # (e.g. ShiftExplainer), so it is trained once
        providers = [method for method in self._methods if hasattr(method, "get_source_error_model_training_mask")]
        for method in self._methods:
            if hasattr(method, "set_source_error_model_provider") and providers:
                method.set_source_error_model_provider(providers[0])

    def get_source_error_model(self) -> Optional[Any]:
        """
        Source error model used by the methods (e.g. ShiftExplainer), to reuse it in a later run. None when no
        method uses one.
        """
        for method in self._methods:
            if hasattr(method, "get_source_error_model"):
                source_error_model = method.get_source_error_model()
                if source_error_model is not None:
                    return source_error_model
        return None

    def get_shift_values(self, x: np.ndarray) -> Optional[Dict[str, Optional[np.ndarray]]]:
        """
        Per-patient shift values of the target observations `x` (see ShiftExplainer.get_shift_values), from the first
        method providing them. None when no method provides them.
        """
        for method in self._methods:
            if hasattr(method, "get_shift_values"):
                shift_values = method.get_shift_values(x)
                if shift_values is not None:
                    return shift_values
        return None

    def context_is_set(self) -> bool:
        """

        """
        for method in self._methods:
            if not method.context_is_set():
                return False
        return True
