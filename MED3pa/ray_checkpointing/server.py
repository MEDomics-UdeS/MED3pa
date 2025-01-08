import ray
from checkpointer import Checkpointer


# Define the server task as an actor
@ray.remote(resources={'head': 1})
class Server:

    def __init__(self, path_provider=None, root_path='checkpoints', verbosity=False, format_chkpt="pickle"):
        self.path_provider = path_provider
        self.verbosity = verbosity
        self.chkpt = Checkpointer(root_path=root_path, verbosity=verbosity, format=format_chkpt)(self.execute)
        self.root_path = self.chkpt.checkpointer.root_path

    def get_settings(self):
        return {'path': self.path_provider, 'root_path': self.root_path, 'verbosity': self.verbosity}

    def execute(self, method, *args, **kwargs):
        result = getattr(self.chkpt.storage, method)(*args, **kwargs)
        return result
