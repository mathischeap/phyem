# -*- coding: utf-8 -*-
r"""
mpiexec -n 1 python tests/msehtt/main.py
mpiexec -n 5 python tests/msehtt/main.py
"""
import sys
if './' not in sys.path:
    sys.path.append('./')

from tools.miscellaneous.ranking import Ranking

import os
source_dir = os.path.dirname(__file__)
ranking = Ranking('msehtt', source_dir + '/__record__.txt')
ranking.start_ranking()
# ----------------- run the tests --------------------------------------------------

from __init__ import php

php('>>> msehtt >>> tests.msehtt.numbering')
from tests.msehtt.numbering import ph_test
for i in range(7):
    ph_test(i)

php('>>> msehtt >>> tests.msehtt.solvers')
# noinspection PyUnresolvedReferences
import tests.msehtt.solvers

php('>>> msehtt >>> tests.msehtt.msepy2_base')
# noinspection PyUnresolvedReferences
import tests.msehtt.msepy2_base

php('>>> msehtt >>> tests.msehtt.msepy3_base')
# noinspection PyUnresolvedReferences
import tests.msehtt.msepy3_base

php('>>> msehtt >>> tests.msehtt.msepy3_base_curvilinear')
# noinspection PyUnresolvedReferences
import tests.msehtt.msepy3_base_curvilinear

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

php('>>> msehtt >>> tests.msehtt.Poisson2di_UnstructuredQuad')
# noinspection PyUnresolvedReferences
import tests.msehtt.Poisson2di_UnstructuredQuad

php('>>> msehtt >>> tests.msehtt.Poisson2do_UnstructuredQuad')
# noinspection PyUnresolvedReferences
import tests.msehtt.Poisson2do_UnstructuredQuad

php('>>> msehtt >>> tests.msehtt._2d_boundary_section_config_1')
# noinspection PyUnresolvedReferences
import tests.msehtt._2d_boundary_section_config_1

php('>>> msehtt >>> tests.msehtt.OTV')
# noinspection PyUnresolvedReferences
import tests.msehtt.OTV

php('>>> msehtt >>> tests.msehtt.Poisson2_outer_meshpy_ts1')
# noinspection PyUnresolvedReferences
import tests.msehtt.Poisson2_outer_meshpy_ts1

php('>>> msehtt >>> tests.msehtt.tsf3_save_read_reduce_tests')
# noinspection PyUnresolvedReferences
import tests.msehtt.tsf3_save_read_reduce_tests


# ------------------------------------------------------------------------------------------


php('>>> msehtt >>> tests.msehtt.adaptive.base2')
# noinspection PyUnresolvedReferences
import tests.msehtt.adaptive.base2

php('>>> msehtt >>> tests.msehtt.adaptive.Poisson2')
# noinspection PyUnresolvedReferences
import tests.msehtt.adaptive.Poisson2

php('>>> msehtt >>> tests.msehtt.adaptive.trf_tests')
# noinspection PyUnresolvedReferences
import tests.msehtt.adaptive.trf_tests


import numpy as np


def refining_function(x, y):
    r""""""
    return np.sin(np.pi * x) * np.sin(np.pi * y)


# php('>>> msehtt >>> tests.msehtt.adaptive.LDC')
# # noinspection PyUnresolvedReferences
# import tests.msehtt.adaptive.LDC


# -------- update records ------------------------------------------------------------------------
ranking.report_ranking()
