# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from tools.frozen import Frozen
from tools.quadrature import quadrature

from src.config import COMM, MPI

from msehtt.static.form.addons.ic import MseHtt_From_InterpolateCopy

from msehtt.static.form.export.main import MseHtt_Static_Form_Export
from msehtt.static.form.project.main import MseHtt_Static_Form_Project

from msehtt.static.space.error.Lambda.Er_m3n3k1 import error__m3n3k1
from msehtt.static.space.error.Lambda.Er_m3n3k2 import error__m3n3k2
from msehtt.static.space.error.Lambda.Er_m3n3k3 import error__m3n3k3
from msehtt.static.space.error.Lambda.Er_m2n2k1 import error__m2n2k1_inner, error__m2n2k1_outer
from msehtt.static.space.error.Lambda.Er_m2n2k2 import error__m2n2k2


class MseHttFormStaticCopy(Frozen):
    """"""

    def __init__(self, f, t):
        """"""
        self._f = f
        self._t = t
        self._field = None
        self._freeze()

    def __repr__(self):
        super_repr = super().__repr__().split('object')[1]
        return self._f.__repr__().split('at')[0] + f'@ {self._t}' + super_repr

    @property
    def degree(self):
        """The degree of the general form."""
        return self._f.degree

    @property
    def tpm(self):
        return self._f.tpm

    @property
    def tgm(self):
        return self._f.tgm

    @property
    def cochain(self):
        return self._f.cochain[self._t]

    @cochain.setter
    def cochain(self, cc):
        """"""
        if cc.__class__ is MseHtt_From_InterpolateCopy:
            # we can direct take the cochain of an interpolation form.
            cc = cc.cochain
        else:
            pass
        self._f.cochain._set(self._t, cc)

    def coboundary(self):
        """Return the coboundary of self's cochain, it is a dict of local cochain vector."""
        return self.cochain.coboundary()

    def reduce(self):
        """"""
        self.cochain = self._f.reduce(self.cf)

    def reconstruct(self, *meshgrid, ravel=False):
        return self._f.reconstruct(self.cochain, *meshgrid, ravel=ravel)

    @property
    def cf(self):
        return self._f.cf[self._t]

    def error(self, error_type='L2'):
        """"""
        if error_type == 'L2':
            return self._f.error(self.cf, self.cochain, error_type=error_type)

        elif error_type == 'd_L2':   # the L2-error of d(self).
            d_cf = self._f.cf.exterior_derivative()[self._t]
            d_cochain = self.cochain.coboundary()
            d_space_str_indicator = self._f.space.d_space_str_indicator

            if d_space_str_indicator == 'm3n3k1':
                d_L2_error = error__m3n3k1(self.tpm, d_cf, d_cochain, self._f.degree, error_type='L2')
            elif d_space_str_indicator == 'm3n3k2':
                d_L2_error = error__m3n3k2(self.tpm, d_cf, d_cochain, self._f.degree, error_type='L2')
            elif d_space_str_indicator == 'm3n3k3':
                d_L2_error = error__m3n3k3(self.tpm, d_cf, d_cochain, self._f.degree, error_type='L2')
            elif d_space_str_indicator == 'm2n2k1_inner':
                d_L2_error = error__m2n2k1_inner(self.tpm, d_cf, d_cochain, self._f.degree, error_type='L2')
            elif d_space_str_indicator == 'm2n2k1_outer':
                d_L2_error = error__m2n2k1_outer(self.tpm, d_cf, d_cochain, self._f.degree, error_type='L2')
            elif d_space_str_indicator == 'm2n2k2':
                d_L2_error = error__m2n2k2(self.tpm, d_cf, d_cochain, self._f.degree, error_type='L2')
            else:
                raise NotImplementedError()

            return float(d_L2_error)

        elif error_type == 'H1':
            self_L2_error = self.error(error_type='L2')
            d_L2_error = self.error(error_type='d_L2')
            return float((self_L2_error ** 2 + d_L2_error ** 2) ** 0.5)

        else:
            raise NotImplementedError(f"error_type={error_type} not implemented.")

    def flux_over_boundary_section(self, boundary_section, quad_degree=5):
        r"""

        Parameters
        ----------
        boundary_section
        quad_degree

        Returns
        -------
        GLOBAL_FLUX:
            The flux over the whole boundary section. We return the same global value in all ranks.

        """
        from msehtt.static.mesh.partial.boundary_section.main import MseHttBoundarySectionPartialMesh
        if boundary_section.__class__ is MseHttBoundarySectionPartialMesh:
            pass
        elif hasattr(boundary_section, 'composition'):

            if boundary_section.composition.__class__ is MseHttBoundarySectionPartialMesh:
                boundary_section = boundary_section.composition
            else:
                raise Exception()
        else:
            raise Exception()

        space = self._f.space

        mn = (space.m, space.n)

        if mn == (2, 2):
            quad = quadrature(quad_degree, 'Gauss')
            quad_nodes, quad_weights = quad.quad

            RANK_FLUX = 0
            for face_id in boundary_section:
                element_index, face_index = face_id
                _, _, U, V = space.RoF(
                    self.degree, self.cochain[element_index], element_index, face_index, quad_nodes
                )
                face = boundary_section[face_id]
                x, y = face.ct.outward_unit_normal_vector(quad_nodes)
                node_flux = U * x + y * V
                JM = face.ct.Jacobian_matrix(quad_nodes)
                Jacobian = np.sqrt(JM[0] ** 2 + JM[1] ** 2)
                flux = sum(node_flux * quad_weights * Jacobian)
                RANK_FLUX += flux
            GLOBAL_FLUX = COMM.allreduce(RANK_FLUX, op=MPI.SUM)
            return GLOBAL_FLUX

        else:
            raise NotImplementedError(f"flux_over_boundary_section not implemented for mn={mn}.")

    @property
    def visualize(self):
        """"""
        return self._f.visualize(self._t)

    def norm(self, norm_type='L2'):
        """

        Parameters
        ----------
        norm_type :
            ``L2_norm``: ((self, self)_{tpm}) ** 0.5

        Returns
        -------

        """
        return float(self._f.norm(self.cochain, norm_type=norm_type))

    def inner_product(self, other, inner_type='L2'):
        r""""""
        return self._f.inner_product(
            self.cochain, other._f, other.degree, other.cochain, inner_type=inner_type,
        )

    def __eq__(self, other):
        r""""""
        if other.__class__ is not self.__class__:
            return False
        else:
            return (self._f is other._f) and (self._t == other._t)

    @property
    def export(self):
        r""""""
        return MseHtt_Static_Form_Export(self._f, self._t)

    @property
    def project(self):
        r""""""
        return MseHtt_Static_Form_Project(self._f, self._t)

    @property
    def numeric(self):
        r""""""
        return ___MseHtt_Static_Form_Copy_Numeric___(self._f, self._t)


class ___MseHtt_Static_Form_Copy_Numeric___(Frozen):
    """"""
    def __init__(self, f, t):
        """"""
        self._f = f
        self._t = t
        self._freeze()

    def rws(self, ddf=1, component_wise=False, **kwargs):
        return self._f.numeric.rws(self._t, ddf=ddf, component_wise=component_wise, **kwargs)

    @property
    def dtype(self):
        return self._f.numeric.dtype

    def interpolate(self, ddf=1, data_only=False, component_wise=False):
        return self._f.numeric._interpolate_(self._t, ddf=ddf, data_only=data_only, component_wise=component_wise)

    def value(self, *coo):
        r"""Find the value of the form at this coordinate."""
        elements = self._f.space.tpm.composition
        in_elements_indexed = elements._find_in_which_elements_(*coo)
        the_element_index = in_elements_indexed[0]
        if the_element_index in elements:
            # we only make the interpolator in one RANK.
            dtype, itp = self.interpolate(component_wise=True)
            if dtype == '2d-scalar':
                w = itp[0]
                x, y = coo
                value = w(x, y)
            else:
                raise NotImplementedError(dtype)

        else:
            value = 0

        return COMM.allreduce(value, op=MPI.SUM)
