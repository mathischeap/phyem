# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.msepy.form.main import MsePyRootForm
from phyem.src.spaces.main import _degree_str_maker
from phyem.tools.quadrature import Quadrature
from phyem.src.spaces.continuous.Lambda import ScalarValuedFormSpace
from phyem.msepy.form.addons.noc.diff_3_entries__BASE import Base3Entries

_3d_data_cache_convect = {}


class _A_convect_B_ip_C(Base3Entries):
    """"""

    def __init__(self, A, B, C, quad=None):
        """(AxB, C)"""
        super().__init__()
        assert A.mesh is B.mesh and A.mesh is C.mesh, f"Meshes do not match!"
        cache_key = list()
        for msepy_form in (A, B, C):
            assert msepy_form.__class__ is MsePyRootForm, f"{msepy_form} is not a {MsePyRootForm}!"
            cache_key.append(
                msepy_form.space.__repr__() + '@degree:' + _degree_str_maker(msepy_form.degree)
            )
        cache_key = ' <=> '.join(cache_key)
        self._cache_key = cache_key

        self._A = A
        self._B = B
        self._C = C
        self._ABC = (A, B, C)
        if all([_.space.abstract.__class__ is ScalarValuedFormSpace for _ in self._ABC]):
            self._type = 'scalar'
        else:
            raise NotImplementedError()

        self._quad = quad
        self._3d_data = None
        self._mesh = A.mesh
        self._e2c = A.mesh.elements._index_mapping._e2c
        self._freeze()

    def _make_3d_data(self):
        """"""
        if self._3d_data is not None:
            return
        else:
            pass

        if self._cache_key in _3d_data_cache_convect:
            self._3d_data = _3d_data_cache_convect[self._cache_key]
            return

        if self._quad is None:
            degrees = list()
            for form in self._ABC:
                degrees.append(
                    form.space[form._degree].p
                )
            degrees = np.array(degrees)
            degrees = np.max(degrees, axis=0)
            degrees = [int(_ * 1.5) + 1 for _ in degrees]
            types = 'Gauss'
        else:
            raise NotImplementedError()

        quad_degrees, quad_types = degrees, types

        quad = Quadrature(quad_degrees, category=types)
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel

        rmA = self._A.reconstruction_matrix(*quad_nodes)
        rmB = self._B.reconstruction_matrix(*quad_nodes)
        rmC = self._C.reconstruction_matrix(*quad_nodes)

        if self._mesh.n == 2:
            xi, et = np.meshgrid(*quad_nodes, indexing='ij')
            xi = xi.ravel('F')
            et = et.ravel('F')
            detJ = self._mesh.elements.ct.Jacobian(xi, et)
        elif self._mesh.n == 3:
            xi, et, sg = np.meshgrid(*quad_nodes, indexing='ij')
            xi = xi.ravel('F')
            et = et.ravel('F')
            sg = sg.ravel('F')
            detJ = self._mesh.elements.ct.Jacobian(xi, et, sg)
        else:
            raise Exception()

        raise NotImplementedError()
