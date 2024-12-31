from ray.util.state import list_tasks
import time


def wait_for_pending_tasks(threshold=600, check_interval=10):
    """
    Wait until the number of pending tasks on the ray cluster is lower than `threshold`.
    It prevents sending too many tasks at the same time on the cluster. If too many tasks are pending,
    wait for `check_interval` seconds before evaluating the number of pending tasks.
    """
    time.sleep(check_interval)  # Initial sleep to ensure enough time between each new tasks
    while True:
        pending_count = len(list_tasks(filters=[('state', '=', 'PENDING_NODE_ASSIGNMENT')],
                                       raise_on_missing_output=False, limit=800))
        if pending_count < threshold:
            break
        time.sleep(check_interval)  # Wait for the next check
