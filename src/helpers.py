import time
import os
import datetime


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


def clear_logs_folder():
    """
        Clear the output directories such
        as output/ and logs/

    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(script_dir, '../logs/')
    for the_file in folder:
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


def get_model_timestamp():
    """
        Returns a timestamp string formatted for
        file names
    Returns:
        str: Timestamp string
    """
    ts = time.time()
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%Hh%M')
