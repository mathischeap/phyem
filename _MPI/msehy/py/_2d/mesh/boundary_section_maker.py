# -*- coding: utf-8 -*-
r"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
"""
from src.config import RANK, MASTER_RANK
from tools.frozen import Frozen
from generic.py._2d_unstruct.mesh.boundary_section.main import BoundarySection
from _MPI.generic.py._2d_unstruct.mesh.boundary_section.main import MPI_Py_2D_Unstructured_BoundarySection


class Generic_BoundarySection_Maker(Frozen):
    """"""

    def __init__(self, serial_generic_bs):
        if RANK == MASTER_RANK:
            assert serial_generic_bs.__class__ is BoundarySection, \
                f"Must have a serial generic boundary section in the master"
        else:
            assert serial_generic_bs is None, f"we must receive None in non-master cores."
        self._sg_bs = serial_generic_bs
        self._freeze()

    def __call__(self):
        """"""
        return MPI_Py_2D_Unstructured_BoundarySection()
