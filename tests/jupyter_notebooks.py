# -*- coding: utf-8 -*-
r"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 7:36 PM on 5/2/2023

$ jupyter nbconvert --to notebook --inplace --execute  ./web/source/jupyter/test1.ipynb

Here we run all jupyter notebooks.

"""

import os
import sys

if './' not in sys.path:
    sys.path.append('./')
from src.config import RANK, MASTER_RANK

if RANK == MASTER_RANK:  # this is conducted only in one rank.

    jupyter_files = [
        r"./web/source/jupyter/general/first_equation",
        r"./web/source/jupyter/general/discretize_linear_pH_system",
    ]

    for jf in jupyter_files:
        stream = os.popen(rf'jupyter nbconvert --to notebook --inplace --execute {jf}.ipynb')
        output = stream.read()
        print(jf, ' output: \n', output)


if __name__ == '__main__':
    # python tests/jupyter_notebooks.py
    pass
