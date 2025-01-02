from checkpointer import get_function_hash
import relib.hashing as hashing


class PathProvider:
    def __init__(self, fn, fn_id, ignored_types=None):
        self.fn_hash, _ = get_function_hash(fn, False)
        self.fn_id = fn_id
        if ignored_types is None:
            ignored_types = []
        elif isinstance(ignored_types, list) or isinstance(ignored_types, tuple):
            self.ignored_types = ignored_types
        else:
            self.ignored_types = [ignored_types]

    def __call__(self, *args, **kwargs):
        args_processed = self.preprocess_hashing(args)
        kwargs_processed = self.preprocess_hashing(kwargs)
        call_hash = hashing.hash((self.fn_hash, args_processed, kwargs_processed), "blake2b")[: 32]
        return f"{self.fn_id}/{call_hash}"

    def preprocess_hashing(self, data):
        """
            Preprocess data to exclude objects of specified types.

            Args:
                data: The data to preprocess (list, tuple, or dict).
                ignored_types: A tuple of types to ignore.

            Returns:
                The preprocessed data with ignored types removed.
            """
        if isinstance(data, tuple):
            return tuple(item for item in data if not type(item) in self.ignored_types)
        elif isinstance(data, list):
            return [item for item in data if not type(item) in self.ignored_types]
        elif isinstance(data, dict):
            return {key: value for key, value in data.items() if not type(value) in self.ignored_types}
        return data
