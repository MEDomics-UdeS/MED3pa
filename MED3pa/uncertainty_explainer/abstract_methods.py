"""

"""
from abc import ABC, abstractmethod
from checkpointer import checkpoint
from checkpointer.function_body import get_fn_body, get_function_hash
import relib.hashing as hashing
from typing import Any, Optional
import numpy as np

from MED3pa.datasets import MaskedDataset

class ExplanationMethod(ABC):
    _reference_dataset = None
    _f_model = None
    # Methods setting this to True receive the target labels y_q in _call__impl
    uses_target_labels = False
    # Instance attributes left out of the checkpoint key (e.g. models fitted lazily from the rest of the state)
    _cache_exclude: tuple = ()

    def __call__(self, x_q:np.ndarray, profile_path:Optional[str] = None, y_q:Optional[np.ndarray] = None):
        if not self.context_is_set():
            raise ValueError("Reference data and base model need to be set before applying the Explainer.")
        call_kwargs = dict(x_q=x_q, profile_path=profile_path)
        if self.uses_target_labels:
            call_kwargs["y_q"] = y_q
        # checkpointer wrapper, keyed on the method's code, its state (settings, reference data, model) and the call
        checkpoint_id = self._checkpoint_id(**call_kwargs)
        wrapped = checkpoint(root_path="checkpoints", verbosity=False,
                             path=lambda *args, **kwargs: checkpoint_id)(self._call__impl)
        return wrapped(**call_kwargs)  # self._call__impl(**call_kwargs)

    def _checkpoint_id(self, **call_kwargs) -> str:
        """
        Checkpoint id of a call. Unlike checkpointer's default id (code and call arguments only), it also depends on
        the method's state, so changing a setting, the reference dataset or the model does not return stale results.
        """
        fn_hash, _ = get_function_hash(self._call__impl, False)
        state_hash = {name: self.__hash_value(value) for name, value in sorted(vars(self).items())
                      if name not in self._cache_exclude}
        call_hash = hashing.hash((fn_hash, state_hash, call_kwargs), "blake2b")[:32]
        return f"{self.get_name()}/{call_hash}"

    @staticmethod
    def __hash_value(value: Any) -> str:
        try:
            return hashing.hash(value, "blake2b")
        except Exception:
            # e.g. a lambda: hash its source code
            if callable(value):
                try:
                    return hashing.hash((getattr(value, "__qualname__", ""), get_fn_body(value)), "blake2b")
                except Exception:
                    pass
            # never matches across instances: no caching rather than stale results
            return f"{type(value).__qualname__}@{id(value)}"

    @abstractmethod
    def _call__impl(self, x_q:np.ndarray, profile_path:Optional[str] = None):
        # Methods with uses_target_labels = True also take y_q:Optional[np.ndarray] = None
        raise NotImplementedError

    def set_context(self, reference_dataset: MaskedDataset, f_model: Any):
        self._reference_dataset = reference_dataset
        self._f_model = f_model

    def context_is_set(self):
        return self._reference_dataset is not None and self._f_model is not None

    def get_name(self):
        return self.__class__.__name__

class CovariateShiftDetector(ExplanationMethod, ABC):
    category = "covariate_shift"

class ConceptShiftDetector(ExplanationMethod, ABC):
    category = "concept_shift"

class CovariateConceptShiftDetector(ExplanationMethod, ABC):
    category = "covariate_and_concept_shift"

class LabelShiftDetector(ExplanationMethod, ABC):
    category = "label_shift"

class UncertaintyEstimator(ExplanationMethod, ABC):
    category = "uncertainty"
