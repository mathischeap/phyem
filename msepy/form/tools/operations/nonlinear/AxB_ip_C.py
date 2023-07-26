# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 5:40 PM on 7/26/2023
"""
from tools.frozen import Frozen
from msepy.form.main import MsePyRootForm
from src.spaces.main import _degree_str_maker

class _AxBipC(Frozen):
    """"""

    def __init__(self, A, B, C, quad=None):
        """(AxB, C)"""
        assert A.mesh is B.mesh and A.mesh is C.mesh, f"Meshes do not match!"
        cache_key = list()
        for msepy_form in (A, B, C):
            assert msepy_form.__class__ is MsePyRootForm, f"{msepy_form} is not a {MsePyRootForm}!"
            cache_key.append(
                msepy_form.__repr__() + '@degree:' + _degree_str_maker(msepy_form.degree)
            )
        cache_key = ' <=> '.join(cache_key)
        self._cache_key = cache_key
        self._A = A
        self._B = B
        self._C = C
        self._quad = quad
        self._3d_data = None
        self._freeze()

    def _make_3d_data(self):
        """"""
        if self._quad is None:
            # TODO:
            pass
        else:
            quad = self._quad

        quad_degrees, quad_types = self._quad



    def __call__(self, dimensions, *args, **kwargs):
        """"""
        if self._3d_data is None:
            self._make_3d_data()
        else:
            pass

        if dimensions == 2:

            row_form, col_form = args

            return self._2d_matrix_representation(row_form, col_form)

        else:
            raise NotImplementedError()

    def _2d_matrix_representation(self, row_form, col_form):
        """We return a dynamic 2D matrix; in each matrix (for a local element), we have a 2d matrix
        whose row indices represent the basis functions of ``row_form`` and whose col indices represent
        the basis functions of ``col_form``.

        Parameters
        ----------
        row_form
        col_form

        Returns
        -------

        """
