from checkpointer import get_function_hash
import relib.hashing as hashing


class PathProvider:
    def __init__(self, fn, fn_id):
        self.fn_hash, _ = get_function_hash(fn, False)
        self.fn_id = fn_id

    def __call__(self, *args, **kwargs):
        print(f'fn_hash: {self.fn_hash} & {args=} & {kwargs=}')
        kwargs2 = kwargs.copy()
        kwargs2.pop('ray_tqdm_bar')
        print(f'fn_hashed: {hashing.hash((self.fn_hash), "blake2b")[: 32]} & '
              f'args {hashing.hash((args), "blake2b")[: 32]} & '
              f'kwargs {hashing.hash((kwargs), "blake2b")[: 32]} &'
              f'kwargs no tqdm {hashing.hash((kwargs2), "blake2b")[: 32]}')
        call_hash = hashing.hash((self.fn_hash, args, kwargs), "blake2b")[: 32]
        return f"{self.fn_id}/{call_hash}"
