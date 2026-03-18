import logging
import time
from typing import Callable


def time_it(name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logging.debug(f"{name} took {end_time - start_time:.2f} seconds")
            return result

        return wrapper

    return decorator
