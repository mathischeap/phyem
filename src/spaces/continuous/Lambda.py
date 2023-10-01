# -*- coding: utf-8 -*-
r"""
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
        Lambda = r"\Lambda"
        if orientation == 'outer':
            self._sym_repr = rf"\widetilde{Lambda}" + r"^{(" + str(self.k) + r')}' + rf"({self.manifold._sym_repr})"
        else:
            self._sym_repr = rf"{Lambda}" + r"^{(" + str(self.k) + r')}' + rf"({self.manifold._sym_repr})"
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
        return f'<{self.orientation} {self.k}-SV on {self.manifold._sym_repr}>'
    
    @property
    def _pure_lin_repr(self):
        """Uniquely representing a space."""
        return f":Lambda:m{self.m}-n{self.n}-k{self.k}-{self.orientation}"
