# -*- coding: utf-8 -*-
from time import localtime, strftime, time


def timeit1(method):
    """A timer decorator for functions or methods."""

    def timed(*args, **kwargs):
        print(" <TimeIt> : Method [%r] with inputs: (%r, %r)" % (method.__name__, args, kwargs))
        print("            started at " + strftime("%Y-%m-%d %H:%M:%S", localtime()))
        ts = time()
        result = method(*args, **kwargs)
        minutes, seconds = divmod(time() - ts, 60)
        hours, minutes = divmod(minutes, 60)
        print(" <TimeIt> : Method [%r] Done, costs： %d:%02d:%02d (hh:mm:ss)" % (
            method.__name__, hours, minutes, seconds))
        return result

    return timed
