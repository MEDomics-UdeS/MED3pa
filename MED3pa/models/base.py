"""This module introduces a singleton manager that manages the instantiation and cloning of a base model,
which is particularly useful for applications like ``med3pa`` where a consistent reference model is
necessary. It employs the **Singleton and Prototype** design patterns to ensure that the base model is instantiated
once and can be cloned without reinitialization."""
from __future__ import annotations

import pickle
from io import BytesIO
from typing import Any, Dict, Optional

from .abstract_models import Model


class BaseModelManager:
    """
    Singleton manager class for the base model. ensures the base model is set only once.
    """
    __baseModel = None
    _threshold = 0.5

    def __init__(self, model: Optional[Model | Any] = None):
        """
        Initializes the BaseModelManager instance.

        Args:
            model (Optional[Model | Any]): The base model to be used.
        """
        self.set_base_model(model)

    def set_base_model(self, model: Model | Any):
        """
        Sets the base model for the manager, ensuring Singleton behavior.
        
        Parameters:
            model (Model | Any): The model to be set as the base model.
            
        Raises:
            TypeError: If the base model has already been initialized.
        """
        if self.__baseModel is None:
            self.__baseModel = model
        else:
            raise TypeError("The Base Model has already been initialized")

    def get_instance(self) -> Model:
        """
        Returns the instance of the base model, ensuring Singleton access.
        
        Returns:
            The base model instance.
            
        Raises:
            TypeError: If the base model has not been initialized yet.
        """
        if self.__baseModel is None:
            raise TypeError("The Base Model has not been initialized yet")
        return self.__baseModel

    def clone_base_model(self) -> Model:
        """
        Creates and returns a deep clone of the base model, following the Prototype pattern.
        
        This method uses serialization and deserialization to clone complex model attributes,
        allowing for independent modification of the cloned model.
        
        Returns:
            A cloned instance of the base model.

        Raises:
            TypeError: If the base model has not been initialized yet.
        """
        if self.__baseModel is None:
            raise TypeError("The Base Model has not been initialized and cannot be cloned")
        else:
            cloned_model = type(self.__baseModel)()
            # Serialize and deserialize the entire base model to create a deep clone.
            if hasattr(self.__baseModel, 'model') and self.__baseModel.model is not None:
                buffer = BytesIO()
                pickle.dump(self.__baseModel.model, buffer)
                buffer.seek(0)
                cloned_model.model = pickle.load(buffer)
                cloned_model.model_class = self.__baseModel.model_class
                cloned_model.pickled_model = True
                cloned_model.params = self.__baseModel.params
            else:
                for attribute, value in vars(self.__baseModel).items():
                    setattr(cloned_model, attribute, value)

            return cloned_model

    def reset(self) -> None:
        """
        Resets the singleton instance, allowing for reinitialization.
        
        This method clears the current base model, enabling the set_base_model method
        to set a new base model.
        """
        self.__baseModel = None

    @property
    def threshold(self):
        if hasattr(BaseModelManager.__getattribute__(self, "_BaseModelManager__baseModel"), "threshold"):
            return self.__baseModel.threshold
        return self._threshold

    def get_info(self) -> Dict[str, Any]:
        """
        Retrieves detailed information about the model.

        Returns:
            Dict[str, Any]: A dictionary containing information about the model's type, parameters,
                            data preparation strategy, and whether it's a pickled model.
        """
        if callable(getattr(self.__baseModel, "get_info", None)):
            return self.__baseModel.get_info()

        return {
            "model": self.__baseModel.__class__.__name__,
            "model_type": self.__baseModel.__class__.__name__,
            "params": self.__baseModel.get_params(),
            "data_preparation_strategy": None,
            "pickled_model": getattr(self.__baseModel, "pickled_model", False),
            "file_path": getattr(self.__baseModel, "file_path", "")
        }

    def __getattr__(self, name) -> Any:
        """
        Returns attributes of the baseModel instance.

        Returns:
            Any: The attributes of the baseModel instance.
        """
        # Delegate attribute access to self.model
        return getattr(self.__baseModel, name)
