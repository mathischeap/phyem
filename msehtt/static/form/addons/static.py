# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen

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

        elif error_type == 'H1':
            self_L2_error = self.error(error_type='L2')
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

            return (self_L2_error ** 2 + d_L2_error ** 2) ** 0.5

        else:
            raise NotImplementedError(f"error_type={error_type} not implemented.")

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
        return self._f.norm(self.cochain, norm_type=norm_type)

    def __eq__(self, other):
        """"""
        if other.__class__ is not self.__class__:
            return False
        else:
            return (self._f is other._f) and (self._t == other._t)

    @property
    def export(self):
        return MseHtt_Static_Form_Export(self._f, self._t)

    @property
    def project(self):
        return MseHtt_Static_Form_Project(self._f, self._t)

    @property
    def numeric(self):
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
