# -*- coding: utf-8 -*-
"""
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

jupyter_path = r"./web/source/jupyter"

jupyter_files = [
    'test1', 'test2', 'test3',
]

for jf in jupyter_files:
    stream = os.popen(rf'jupyter nbconvert --to notebook --inplace --execute {jupyter_path}/{jf}.ipynb')
    output = stream.read()
    print(jf, ' output: \n', output)


if __name__ == '__main__':
    # python tests/jupyter_notebooks.py
    pass
