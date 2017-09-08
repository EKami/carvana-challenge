import time


def st_time(show_func_name=True):
    """
        Decorator to calculate the total time of a func

    Args:
        show_func_name (bool): Whether to show the function name or not
    """

    def wrapper(func):
        def st_func(*args, **keyArgs):
            t1 = time.time()
            r = func(*args, **keyArgs)
            t2 = time.time()
            if show_func_name:
                print("Function=%s, Time elapsed = %ds" % (func.__name__, t2 - t1))
            else:
                print("Time elapsed = %ds" % (t2 - t1))
            return r

        return st_func
    return wrapper

