# -*- coding: utf-8 -*-
r"""
python tests/msehtt/main.py 5
"""
import subprocess

from phyem.src.config import SIZE
assert SIZE == 1, f"can only use 1 rank for the msehtt-test-kernel. It will call subprocess with MPI."
from phyem import php

print(rf"Starting <<<tools.tests>>> ....")

import os
source_dir = os.path.dirname(__file__)

tasks = [
    "geometries_m2n2",
]

for task in tasks:
    php('>>> tools >>>', task)
    popen = subprocess.Popen(
        rf"python {source_dir}/{task}.py", stdout=subprocess.PIPE, universal_newlines=True)
    for line in popen.stdout:
        print(line[:-1])
    popen.kill()
