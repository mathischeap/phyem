# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
@time: 11/26/2022 2:56 PM
"""

from src.spaces.base import SpaceBase


class ScalarValuedFormSpace(SpaceBase):
    """
    Parameters
    ----------
    mesh :
    k :
        k-form spaces

    Examples
    --------

    """

    def __init__(self, mesh, k, orientation='outer'):
        super().__init__(mesh, orientation)
        assert isinstance(k, int) and 0 <= k <= mesh.ndim, f" k={k} illegal on {mesh}."
        self._k = k
        if orientation == 'outer':
            self._sym_repr = r"\widetilde{\Lambda}^{(" + str(self.k) + r')}' + rf"({self.manifold._sym_repr})"
        else:
            self._sym_repr = r"\Lambda^{(" + str(self.k) + r')}' + rf"({self.manifold._sym_repr})"

        self._freeze()

    @property
    def indicator(self):
        """The string that indicates Scalar valued form spaces."""
        return "Lambda"

    @property
    def k(self):
        """I am k-form."""
        return self._k

    def __repr__(self):
        """By construction, it will be unique."""
        super_repr = super().__repr__().split('object')[-1]
        return f'<{self.orientation}-oriented {self._sym_repr}' + super_repr
    
    @property
    def _pure_lin_repr(self):
        """Uniquely representing a space."""
        return f":Lambda:m{self.m}-n{self.n}-k{self.k}-{self.orientation}"
