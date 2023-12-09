# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from phmpi.generic.py.vector.localize.static import (MPI_PY_Localize_Static_Vector,
                                                     MPI_PY_Localize_Static_Vector_Cochain)


class MPI_PY_Localize_Dynamic_Vector(Frozen):
    """"""

    def __init__(self, localize_static_vector_caller):
        """"""
        assert callable(localize_static_vector_caller), f"vector caller must be callable!"
        self._static_caller = localize_static_vector_caller
        self._freeze()

    def __call__(self, *args, **kwargs):
        static = self._static_caller(*args, **kwargs)
        assert static.__class__ in (MPI_PY_Localize_Static_Vector, MPI_PY_Localize_Static_Vector_Cochain), \
            f"call a dynamic local vector must give a static one. Now, we get {static.__class__}"
        return static


class MPI_PY_Localize_Dynamic_Vector_Cochain(MPI_PY_Localize_Dynamic_Vector):
    """"""

    def __init__(self, dynamic_cochain_caller):
        """"""
        super().__init__(dynamic_cochain_caller)
        self._freeze()
