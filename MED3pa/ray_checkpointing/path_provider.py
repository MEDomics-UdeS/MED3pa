from checkpointer import get_function_hash
import relib.hashing as hashing


class PathProvider:
    def __init__(self, fn, fn_id):
        self.fn_hash, _ = get_function_hash(fn, False)
        self.fn_id = fn_id

    def __call__(self, *args, **kwargs):
        call_hash = hashing.hash((self.fn_hash, args, kwargs), "blake2b")[: 32]
        return f"{self.fn_id}/{call_hash}"
