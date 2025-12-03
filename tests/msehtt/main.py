# -*- coding: utf-8 -*-
r"""
python tests/msehtt/main.py 5
"""
import subprocess
import sys

from phyem.src.config import SIZE
assert SIZE == 1, f"can only use 1 rank for the msehtt-test-kernel. It will call subprocess with MPI."
from phyem import php

if len(sys.argv) == 1:
    use_ranks = 5
else:
    use_ranks = int(sys.argv[1])
    assert 1 < use_ranks <= 12, f"use_ranks={use_ranks} is wrong. It should be a integer in [1, 12]."

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
    popen.kill()

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
    ("adaptive/trf_tests", 5),  # means we can use at most 5 ranks.

    "multigrid/Poisson2",
    "multigrid/Poisson3",
    "multigrid/Poisson2_periodic",
    "multigrid/PNPNS_decoupled_linearized",
]

for task in tasks:
    if isinstance(task, str):
        the_task = task
        task_exec_cmd = exec_cmd
    elif isinstance(task, tuple):  # we provide a task dir and the max amount of ranks it can use.
        the_task = task[0]
        task_max_ranks = task[1]
        if task_max_ranks < use_ranks:
            task_exec_cmd = f"mpiexec -n {task_max_ranks} python "
        else:
            task_exec_cmd = exec_cmd
    else:
        raise NotImplementedError(task)

    php('>>> msehtt >>>', the_task)
    popen = subprocess.Popen(
        task_exec_cmd + rf"{source_dir}/{the_task}.py", stdout=subprocess.PIPE, universal_newlines=True)
    for line in popen.stdout:
        print(line[:-1])
    popen.kill()

# -------- update records ------------------------------------------------------------------------
ranking.report_ranking()
