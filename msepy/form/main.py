# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 1:22 PM on 4/17/2023
"""

import sys

if './' not in sys.path:
    sys.path.append('./')
from src.tools.frozen import Frozen


class MsePyRootForm(Frozen):
    """"""

    def __init__(self, abstract_root_form):
        """"""
        self._abstract = abstract_root_form
        self._freeze()

    @property
    def abstract(self):
        """the abstract object this root-form is for."""
        return self._abstract


if __name__ == '__main__':
    # python msepy/form/main.py
    pass
