# -*- coding: utf-8 -*-
r"""
To run a particular jupyter notebook, do:

$ jupyter nbconvert --to notebook --inplace --execute  ./web/source/jupyter/test1.ipynb

Here we run all jupyter notebooks. Just do

# python tests/jupyter_notebooks.py

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

else:
    pass
