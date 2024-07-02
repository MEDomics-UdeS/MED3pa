"""
This module introduces the ``EarlyStopper`` class, a utility designed to prevent overfitting and reduce computational overhead by halting the training process when improvement in a monitored metric ceases. 
"""
class EarlyStopper:
    """
    A utility class for early stopping, which is used to terminate training processes if certain conditions are met.
    This helps in preventing overfitting and reduces unnecessary training time by stopping the training when 
    the monitored metric has stopped improving.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Initializes the EarlyStopper instance with the specified patience, minimum change (delta), and mode.

        Args:
            patience (int, optional): The number of epochs to wait for improvement. Default is 10.
            min_delta (float, optional): The minimum change in the monitored quantity to qualify as an improvement,
                                         helping to define the sensitivity of the early stopping. Default is 0.0.
            mode (str, optional): The operational mode, either 'min' or 'max'. If 'min', training will stop when the
                                  quantity monitored has stopped decreasing; if 'max', training will stop when the
                                  quantity has stopped increasing. Default is 'min'.

        Raises:
            AssertionError: If the mode provided is not 'min' or 'max'.
        """
        assert mode in ['min', 'max'], "Mode must be 'min' or 'max'"
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.wait = 0
        self.mode = mode

    def update(self, metric: float) -> bool:
        """
        Updates the early stopper with the latest metric and checks whether the stopping condition has been met.

        Args:
            metric (float): The current value of the monitored metric.

        Returns:
            bool: Returns True if the early stopping condition has been met (i.e., no improvement for the specified
                  number of epochs), and False otherwise.
        """
        if self.best is None:
            self.best = metric
            return False

        if self.mode == 'min':
            if metric < self.best - self.min_delta:
                self.best = metric
                self.wait = 0
                return False
            else:
                self.wait += 1
                return self.wait >= self.patience

        elif self.mode == 'max':
            if metric > self.best + self.min_delta:
                self.best = metric
                self.wait = 0
                return False
            else:
                self.wait += 1
                return self.wait >= self.patience
