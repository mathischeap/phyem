# -*- coding: utf-8 -*-
"""
mpiexec -n 1 python tests/msehtt/main.py
mpiexec -n 4 python tests/msehtt/main.py
"""

import sys
if './' not in sys.path:
    sys.path.append('./')

# noinspection PyUnresolvedReferences
import tests.msehtt.solvers

# noinspection PyUnresolvedReferences
import tests.msehtt.msepy2_base

# noinspection PyUnresolvedReferences
import tests.msehtt.Poisson2

# noinspection PyUnresolvedReferences
import tests.msehtt.Poisson2_periodic
