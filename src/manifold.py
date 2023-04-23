# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
created at: 2/20/2023 11:41 AM
"""
import sys

if './' not in sys.path:
    sys.path.append('./')

from tools.frozen import Frozen
from src.config import get_embedding_space_dim
from src.config import _manifold_default_sym_repr
from src.config import _check_sym_repr
from src.config import _parse_lin_repr
from src.config import _manifold_default_lin_repr
from src.config import _manifold_partition_lin_repr

_global_manifolds = dict()  # all manifolds are cached, and all sym_repr and lin_repr are different.


def manifold(
        ndim,
        sym_repr=None,
        lin_repr=None,
        is_periodic=False
):
    """A function wrapper of the Manifold class."""
    return Manifold(ndim, sym_repr=sym_repr, lin_repr=lin_repr, is_periodic=is_periodic)


class Manifold(Frozen):
    """"""

    def __init__(
        self, ndim,
        sym_repr=None,
        lin_repr=None,
        is_periodic=False,
        # add other representations here.
    ):
        """"""
        
        embedding_space_ndim = get_embedding_space_dim()
        assert ndim % 1 == 0 and 0 <= ndim <= embedding_space_ndim, \
            f"manifold ndim={ndim} is wrong. Is should be an integer and be in range [0, {embedding_space_ndim}]. " \
            f"You change change the dimensions of the embedding space using function `config.set_embedding_space_dim`."
        self._ndim = ndim
        if sym_repr is None:
            base_repr = _manifold_default_sym_repr
            number_existing_manifolds = len(_global_manifolds)

            if number_existing_manifolds == 0:
                sym_repr = base_repr
            else:
                sym_repr = base_repr + r'_{' + str(number_existing_manifolds) + '}'
        else:
            pass
        sym_repr = _check_sym_repr(sym_repr)

        if lin_repr is None:
            base_repr = _manifold_default_lin_repr
            number_existing_manifolds = len(_global_manifolds)

            if number_existing_manifolds == 0:
                lin_repr = base_repr
            else:
                lin_repr = base_repr + str(number_existing_manifolds)

        assert sym_repr not in _global_manifolds, \
            f"Manifold symbolic representation is illegal, pls specify a symbolic representation other than " \
            f"{set(_global_manifolds.keys())}"

        for _ in _global_manifolds:
            _m = _global_manifolds[_]
            assert lin_repr != _m._lin_repr
        lin_repr, pure_lin_repr = _parse_lin_repr('manifold', lin_repr)

        self._sym_repr = sym_repr
        self._lin_repr = lin_repr
        self._pure_lin_repr = pure_lin_repr
        _global_manifolds[sym_repr] = self

        assert isinstance(is_periodic, bool), f"is_periodic must be bool type."
        self._is_periodic = is_periodic

        self._udg = None  # if it has an udg_repr representation.
        self._boundary = None
        self._inclusion = None  # not None for boundary manifold. Will be set when initialize a boundary manifold.
        self._sub_manifolds = {  # the sub-manifolds of the same dimensions. Using sym_repr as cache key.
            self._sym_repr: self
        }
        self._partitions = {
            '0': (self, )
        }
        self._covered_by_mesh = None  # if we have generated an abstract mesh for it, return the mesh
        self._freeze()

    @property
    def esd(self):
        """embedding space dimensions"""
        return get_embedding_space_dim()

    @property
    def ndim(self):
        """The dimensions of this manifold."""
        return self._ndim

    @property
    def udg(self):
        """the undirected graph representation of this manifold."""
        return self._udg

    def is_periodic(self):
        """"""
        return self._is_periodic

    def __repr__(self):
        """"""
        super_repr = super().__repr__().split('object')[-1]
        return f'<Manifold {self._sym_repr}' + super_repr  # this must be unique.

    # it is regarded as an operator, do not @property.
    def boundary(self, sym_repr=None):
        """Give a manifold of dimensions (n-1)"""
        if self._boundary is None:
            if self.ndim == 0:
                return NullManifold('Null')
            elif self.is_periodic():
                return NullManifold(self.ndim-1)
            else:
                if sym_repr is None:
                    sym_repr = r'\partial' + self._sym_repr
                self._boundary = Manifold(
                    self.ndim-1,
                    sym_repr=sym_repr,
                    lin_repr=f'boundary-of-{self._pure_lin_repr}',
                    is_periodic=True,
                )
                self._boundary._inclusion = self
        return self._boundary

    def inclusion(self):
        """"""
        return self._inclusion

    def cap(self, other, sym_repr=None):
        """return the intersection of two manifolds, i.e., return manifold := self cap other."""
        raise NotImplementedError()

    def interface(self, other, sym_repr=None):
        """return the cap of boundaries of two manifolds."""
        raise NotImplementedError()

    def partition(self, *submanifolds_sym_repr, config_name=None):
        """M = M1 U M2 U M3 U .... and Mi cap Mj = empty."""
        for sym_repr in submanifolds_sym_repr:
            assert isinstance(sym_repr, str), f"please put sym_repr of partitions in str."
        num_of_partitions = len(submanifolds_sym_repr)
        assert num_of_partitions >= 2 and num_of_partitions % 2 == 0, f"I need a integer >= 2."
        partitions = tuple()

        for i in range(num_of_partitions):
            sym_repr = submanifolds_sym_repr[i]
            if sym_repr in self._sub_manifolds:
                pass
            else:
                j = len(self._sub_manifolds)
                while 1:
                    lin_repr = self._pure_lin_repr + _manifold_partition_lin_repr + f'{str(j)}'
                    occupied = False
                    for _sub_sym_repr in self._sub_manifolds:
                        if lin_repr == self._sub_manifolds[_sub_sym_repr]:
                            occupied = True
                        else:
                            pass
                    if occupied:
                        j += 1
                    else:
                        break

                self._sub_manifolds[sym_repr] = Manifold(
                    self.ndim,
                    sym_repr=sym_repr,
                    lin_repr=lin_repr,
                )
            partitions += (self._sub_manifolds[sym_repr],)

        if config_name is None:
            i = len(self._partitions)
            while 1:
                if str(i) not in self._partitions:
                    break
                else:
                    i += 1
            config_name = str(i)
        else:
            pass

        partitions_set = set(partitions)
        existing = False
        existing_config = None
        for existing_config in self._partitions:
            existing_partitions = set(self._partitions[existing_config])
            if partitions_set == existing_partitions:
                existing = True
                break
        if existing:
            return self._partitions[existing_config]
        else:
            self._partitions[config_name] = partitions
            return partitions

    def _manifold_text(self):
        """generate text for printing representations."""
        return rf'In ${self._sym_repr}\subset\mathbb' + '{R}^{' + str(get_embedding_space_dim()) + '}$, '


class NullManifold(Frozen):
    """"""

    def __init__(self, ndim):
        """"""
        self._ndim = ndim
        self._freeze()

    @property
    def ndim(self):
        """"""
        return self._ndim


if __name__ == '__main__':
    # python src/manifold.py
    import __init__ as ph

    m1 = ph.manifold(3)
    m0 = m1.boundary()
    print(m0, m0.inclusion())
