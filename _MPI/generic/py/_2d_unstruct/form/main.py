# -*- coding: utf-8 -*-
r"""
"""
from generic.py._2d_unstruct.form.main import GenericUnstructuredForm2D

from _MPI.generic.py.cochain.main import MPI_PY_Form_Cochain
from _MPI.generic.py._2d_unstruct.form.visualize import MPI_PY_2D_FORM_VISUALIZE


class MPI_Py_2D_Unstructured_Form(GenericUnstructuredForm2D):
    """"""

    def __repr__(self):
        """repr"""
        return f"<form in {self._space}"

    @property
    def cochain(self):
        """"""
        if self._cochain is None:
            self._cochain = MPI_PY_Form_Cochain(self)
        return self._cochain

    @property
    def visualize(self):
        """visualize"""
        if self._vis is None:
            self._vis = MPI_PY_2D_FORM_VISUALIZE(self)
        return self._vis
