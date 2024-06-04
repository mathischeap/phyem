# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehtt.static.form.export.main import MseHtt_Static_Form_Export
from msehtt.static.space.error.Lambda.Er_m3n3k1 import error__m3n3k1
from msehtt.static.space.error.Lambda.Er_m3n3k2 import error__m3n3k2
from msehtt.static.space.error.Lambda.Er_m3n3k3 import error__m3n3k3


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
        self._f.cochain._set(self._t, cc)

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
