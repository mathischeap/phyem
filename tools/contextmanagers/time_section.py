# -*- coding: utf-8 -*-
r"""
"""

import contextlib
from time import localtime, strftime, time
from src.config import SIZE


@contextlib.contextmanager
def time_section(info=None):
    """"""
    if SIZE == 1:  # single rank computation.
        print("\n <{}> starts at [".format(info) + strftime("%Y-%m-%d %H:%M:%S", localtime()) + ']')
        ts = time()
        yield
        print("    - <{}> ends at [".format(info) + strftime("%Y-%m-%d %H:%M:%S", localtime()) + ']')
        print("    - <{}> costs: [%.5f seconds]\n".format(info) % (time()-ts))
    else:
        raise NotImplementedError()
