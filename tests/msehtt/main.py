# -*- coding: utf-8 -*-
r"""
python tests/msehtt/main.py 5
"""
import subprocess
import sys

from phyem.src.config import SIZE
assert SIZE == 1, f"can only use 1 rank for the test."
from phyem import php

if len(sys.argv) == 1:
    use_ranks = 5
else:
    use_ranks = int(sys.argv[1])
    assert 1 < use_ranks <= 12, f"use_ranks={use_ranks} is wrong."

print(rf"Starting <<<msehtt.tests>>> with {use_ranks} ranks....")
exec_cmd = f"mpiexec -n {use_ranks} python "
from phyem.tools.miscellaneous.ranking import Ranking
import os
source_dir = os.path.dirname(__file__)
ranking = Ranking('msehtt', source_dir + '/__record__.txt', num_ranks=use_ranks)
ranking.start_ranking()

# ----------------- run the tests --------------------------------------------------
php('>>> msehtt >>> numbering')
for i in range(7):
    if i in (4, 5) and use_ranks > 5:
        EXEC_CMD = f"mpiexec -n 5 python "
    else:
        EXEC_CMD = exec_cmd
    popen = subprocess.Popen(
        EXEC_CMD + rf"{source_dir}/numbering.py {i}",
        stdout=subprocess.PIPE, universal_newlines=True
    )
    for line in popen.stdout:
        print(line[:-1])

tasks = [
    "solvers",
    "msepy2_base",
    "msepy3_base",
    "msepy3_base_curvilinear",
    "Poisson2",
    "Poisson3",
    "Poisson2_periodic",
    "Poisson3_periodic",
    "flow_around_cylinder",
    "dMHD_LDC",
    "dMHD_manu",
    "dMHD_manu_cache",

    "OTV",
    "Poisson2di_UnstructuredQuad",
    "Poisson2do_UnstructuredQuad",
    "_2d_boundary_section_config_1",
    "Poisson2_outer_meshpy_ts1",
    "tsf3_save_read_reduce_tests",

    "PNPNS/linearSchemePeriodicManu",
    "PNPNS/linearSchemeBcManu",
    "PNPNS/nonlinearSchemePeriodicManu",

    "adaptive/base2",
    "adaptive/Poisson2",
    "adaptive/trf_tests",
]

for task in tasks:
    php('>>> msehtt >>>', task)
    popen = subprocess.Popen(exec_cmd + rf"{source_dir}/{task}.py", stdout=subprocess.PIPE, universal_newlines=True)
    for line in popen.stdout:
        print(line[:-1])


# -------- update records ------------------------------------------------------------------------
ranking.report_ranking()
