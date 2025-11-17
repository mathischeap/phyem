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

from phyem.src.config import SIZE

__all__ = [
    'msepy',
    'jupyter',
    'web',

    'msehtt',
]

if SIZE == 1:
    import phyem.tests.msepy.main as msepy
    import phyem.tests.jupyter_notebooks as jupyter
    import phyem.tests.web as web

else:
    pass

import phyem.tests.msehtt.main as msehtt
