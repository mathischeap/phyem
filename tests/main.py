# -*- coding: utf-8 -*-
r"""
Do all tests.

By running this file, we
    - do all tests (doctests and unittests)
    - re-compile the all jupyter notebooks
    - re-generate the web page if possible.

$ python tests/main.py
$ mpiexec -n 1 python tests/main.py
$ mpiexec -n 4 python tests/main.py

"""
import sys
if './' not in sys.path:
    sys.path.append('./')

from src.config import SIZE

__all__ = [
    'msepy',
    'jupyter',
    'web',

    'msehtt',
]

if SIZE == 1:
    import tests.msepy.main as msepy
    import tests.jupyter_notebooks as jupyter
    import tests.web as web

else:
    pass

import tests.msehtt.main as msehtt
