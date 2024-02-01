# -*- coding: utf-8 -*-
r"""
Do all tests.

By running this file, we
    - do all tests (doctests and unittests)
    - re-compile the all jupyter notebooks
    - re-generate the web page if possible.

$ python tests/main.py

$ mpiexec -n 4 python tests/main.py

"""
import os
import sys
if './' not in sys.path:
    sys.path.append('./')

from src.config import SIZE

__all__ = [
    'jupyter',
    'web',
]

if SIZE == 1:
    print(  # we use a container to do the test for safety reasons.
        os.popen('python tests/msepy/main.py').read()
    )
    import tests.jupyter_notebooks as jupyter
    import tests.web as web

else:
    pass
