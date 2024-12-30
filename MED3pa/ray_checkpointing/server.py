import ray
from checkpointer import Checkpointer


# Define the server task as an actor
@ray.remote(resources={'head': 1})
class Server:

    def __init__(self, root_path='checkpoints', verbosity=False):
        self.verbosity = verbosity
        self.chkpt = Checkpointer(root_path=root_path, verbosity=verbosity)(self.execute)
        self.root_path = self.chkpt.checkpointer.root_path

    def get_settings(self):
        return {'root_path': self.root_path, 'verbosity': self.verbosity}

    def execute(self, method, *args, **kwargs):
        result = getattr(self.chkpt.storage, method)(*args, **kwargs)
        return result
