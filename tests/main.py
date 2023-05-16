# -*- coding: utf-8 -*-
r"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 7:20 PM on 5/2/2023

By run this file, we
    - do all tests (doctests and unittests)
    - re-compile the all jupyter notebooks
    - re-generate the web page if possible.

$ python tests/main.py

"""

import sys
if './' not in sys.path:
    sys.path.append('./')
from src.config import RANK, MASTER_RANK

__all__ = [
    'unittests',
    'jupyter',
    'web',
]

error_report = list()

import tests.unittests.main as unittests
import tests.jupyter_notebooks as jupyter
import tests.web as web

error_report.extend(
    unittests.unittest_error_report
)

if RANK == MASTER_RANK:
    if len(error_report) != 0:
        print(">>> tests went wrong: \n")
        for i, error_item in enumerate(error_report):
            print(rf"{i}:", error_item, '\n')
