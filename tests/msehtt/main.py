# -*- coding: utf-8 -*-
r"""
mpiexec -n 1 python tests/msehtt/main.py
mpiexec -n 5 python tests/msehtt/main.py
"""

import sys
if './' not in sys.path:
    sys.path.append('./')

from __init__ import php

php('>>> msehtt >>> tests.msehtt.solvers')
# noinspection PyUnresolvedReferences
import tests.msehtt.solvers

php('>>> msehtt >>> tests.msehtt.msepy2_base')
# noinspection PyUnresolvedReferences
import tests.msehtt.msepy2_base

php('>>> msehtt >>> tests.msehtt.msepy3_base')
# noinspection PyUnresolvedReferences
import tests.msehtt.msepy3_base

php('>>> msehtt >>> tests.msehtt.Poisson2')
# noinspection PyUnresolvedReferences
import tests.msehtt.Poisson2

php('>>> msehtt >>> tests.msehtt.Poisson3')
# noinspection PyUnresolvedReferences
import tests.msehtt.Poisson3

php('>>> msehtt >>> tests.msehtt.Poisson2_periodic')
# noinspection PyUnresolvedReferences
import tests.msehtt.Poisson2_periodic

php('>>> msehtt >>> tests.msehtt.Poisson3_periodic')
# noinspection PyUnresolvedReferences
import tests.msehtt.Poisson3_periodic

php('>>> msehtt >>> tests.msehtt.flow_around_cylinder')
# noinspection PyUnresolvedReferences
import tests.msehtt.flow_around_cylinder

php('>>> msehtt >>> tests.msehtt.dMHD_LDC')
# noinspection PyUnresolvedReferences
import tests.msehtt.dMHD_LDC

php('>>> msehtt >>> tests.msehtt.dMHD_manu')
# noinspection PyUnresolvedReferences
import tests.msehtt.dMHD_manu

php('>>> msehtt >>> tests.msehtt.dMHD_manu_cache')
# noinspection PyUnresolvedReferences
import tests.msehtt.dMHD_manu_cache
