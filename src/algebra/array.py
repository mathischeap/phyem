# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
created at: 3/21/2023 2:35 PM
"""
import sys

if './' not in sys.path:
    sys.path.append('./')

import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
})
matplotlib.use('TkAgg')

from tools.frozen import Frozen
from src.config import _parse_lin_repr, _abstract_array_factor_sep, _abstract_array_connector
from src.config import _transpose_text
from src.form.parameters import constant_scalar
_cs1 = constant_scalar(1)


_global_root_arrays = dict()  # using lin_repr as cache keys


def _root_array(sym_repr, pure_lin_repr, shape, symmetric=None, transposed=None):
    """"""
    if pure_lin_repr in _global_root_arrays:
        return _global_root_arrays[pure_lin_repr]   # can provide lin_repr only for existing array.

    lin_repr, pure_lin_repr = _parse_lin_repr('array', pure_lin_repr)
    if lin_repr in _global_root_arrays:
        return _global_root_arrays[lin_repr]
    else:
        ra = AbstractArray(
            is_root=True, sym_repr=sym_repr, lin_repr=lin_repr,
            shape=shape, symmetric=symmetric, transposed=transposed)
        ra._pure_lin_repr = pure_lin_repr
        _global_root_arrays[lin_repr] = ra
        return ra


class AbstractArray(Frozen):
    """"""

    def __init__(
            self,
            is_root=False, sym_repr=None, lin_repr=None,
            shape=None, symmetric=False, transposed=False,

            factor=None, components=None, transposes=None,
    ):
        """"""
        if is_root:
            assert sym_repr is not None and lin_repr is not None, f"give `sym_repr` and `lin_repr` for root array."
            assert factor is None and components is None and transposes is None
            self.___sym_repr___ = sym_repr
            self.___lin_repr___ = lin_repr
            self._components = (self, )
            self._transposes = (False, )
            assert isinstance(shape, tuple), f"pls put shape in a tuple."
            self.___shape___ = shape
        else:
            assert components is not None and transposes is not None, \
                f"pls give components and transposes for non-root-array"
            self.___sym_repr___ = None
            self.___lin_repr___ = None
            self._components = components
            self._transposes = transposes
            self.___shape___ = None

        if factor is None:
            factor = _cs1
        else:
            pass
        self._factor = factor
        self._is_root = is_root
        self._pure_lin_repr = None     # only root array has it.
        self._symmetric = symmetric    # True or False only for 2-d array.
        self._transposed = transposed  # True or False only for 2-d array.
        self._freeze()

    @property
    def _sym_repr(self):
        if self.___sym_repr___ is None:
            if self._factor == _cs1:
                sym = ''
            else:
                if self._factor.is_root():
                    sym = self._factor._sym_repr
                else:
                    sym = r'\left(' + self._factor._sym_repr + r"\right)"

            for i, com in enumerate(self._components):
                csr = com._sym_repr
                trans = self._transposes[i]
                if trans:
                    csr = r"\{" + csr + r"\}^\mathsf{T}"
                else:
                    pass
                sym += csr

            self.___sym_repr___ = sym

        return self.___sym_repr___

    @property
    def _lin_repr(self):
        if self.___lin_repr___ is None:
            if self._factor == _cs1:
                lin = ''
            else:
                lin = self._factor._lin_repr + _abstract_array_factor_sep

            for i, com in enumerate(self._components):
                clr = com._lin_repr
                trans = self._transposes[i]
                if trans:
                    clr += _transpose_text
                else:
                    pass

                if 0 < i:
                    lin += _abstract_array_connector

                lin += clr

            self.___lin_repr___ = lin

        return self.___lin_repr___

    def is_root(self):
        """"""
        return self._is_root

    @property
    def shape(self):
        if self.___shape___ is None:

            s = None
            for i, com in enumerate(self._components):
                shape = com.shape
                trans = self._transposes[i]

                if trans:
                    assert len(shape) == 2
                    shape = (shape[1], shape[0])
                else:
                    pass

                if s is None:
                    s = shape
                else:
                    assert s[-1] == shape[0]

                    s = s[:-1] + shape[1:]

            self.___shape___ = s

        return self.___shape___

    def _shape_text(self):
        if self.ndim == 2 and self.shape[1] == 1:
            return r'\mathbb{R}^{\#_{dof}' + self.shape[0] + r"}"
        else:
            raise NotImplementedError()

    def pr(self, figsize=(8, 6)):
        """print representations."""
        text = r"$" + self._sym_repr + r"$"
        text += "\n" + self._lin_repr
        plt.figure(figsize=figsize)
        plt.axis([0, 1, 0, 1])
        plt.axis('off')
        plt.text(0.05, 0.5, text, ha='left', va='center', size=15)
        plt.tight_layout()
        plt.show()

    def __repr__(self):
        """repr"""
        super_repr = super().__repr__().split('object')[1]
        return f"<AbstractArray {self._lin_repr} of shape {self.shape}" + super_repr

    @property
    def ndim(self):
        return len(self.shape)

    def __len__(self):
        """"""
        return len(self._components)

    def transpose(self):
        """"""
        raise NotImplementedError()

    @property
    def T(self):
        """Transpose."""
        if len(self) == 1:
            assert self.ndim == 2, f"Only 2d array has T property."

            if self._symmetric:
                return self
            else:
                pass

            if self._transposed:

                pure_lin_repr = self._components[0]._pure_lin_repr

                return _root_array(   # must return an existing root-array
                    '',
                    pure_lin_repr,
                    None
                )

            else:
                assert self.is_root(), f"trivial check!"
                return AbstractArray(
                    factor=self._factor,
                    components=self._components,
                    transposes=(True, )
                )

        else:
            raise NotImplementedError(f'transpose not implemented for len={len(self)}.')

    def __matmul__(self, other):
        """self @ other"""
        if other.__class__ == self.__class__:
            if self._factor == _cs1 and other._factor == _cs1:

                return AbstractArray(
                    factor=None,   # _cs1
                    components=self._components + other._components,
                    transposes=self._transposes + other._transposes,
                )

            elif self._factor == _cs1 or other._factor == _cs1:

                if self._factor == _cs1:
                    factor = other._factor
                else:
                    factor = self._factor

                return AbstractArray(
                    factor=factor,   # _cs1
                    components=self._components + other._components,
                    transposes=self._transposes + other._transposes,
                )

            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError(f"{other}")

    def __rmul__(self, other):
        """other * self."""

        if other.__class__.__name__ == "ConstantScalar0Form":

            return AbstractArray(
                factor=other,   # _cs1
                components=self._components,
                transposes=self._transposes,
            )

        else:
            raise NotImplementedError()
