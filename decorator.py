from functools import wraps
from time import time


def FpsPerformance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        cost_time = end - start
        try:
            print(f"func <{func.__name__}> run time {cost_time * 1000}ms, FPS {round(1 / cost_time, 2)}")
        except ZeroDivisionError as e:
            print(f"func <{func.__name__}> run time {cost_time * 1000}ms, FPS MAX")
        return result

    return wrapper


def Singleton(cls):
    _instance = {}

    def _singleton(*args, **kwagrs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwagrs)
        return _instance[cls]

    return _singleton
