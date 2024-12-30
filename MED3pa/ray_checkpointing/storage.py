from pathlib import Path
from checkpointer import Storage
import ray


def get_storage(server):
    class HeadStorage(Storage):
        def __init__(self, *args, **kwargs):
            """
            Initialize the HeadStorage.

            Args:
                head_root (Path): The root directory on the head node where all checkpoints will be stored.
            """
            self.server = server
            super().__init__(*args, **kwargs)

        def __getattribute__(self, method):
            if not callable(super().__getattribute__(method)):
                return super().__getattribute__(method)

            def default(*args, **kwargs):
                result = ray.get(self.server.execute.remote(method, *args, **kwargs))
                return result
            return default

    return HeadStorage
