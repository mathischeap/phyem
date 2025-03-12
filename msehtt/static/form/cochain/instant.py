# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from tools.frozen import Frozen
from src.config import COMM, MASTER_RANK, RANK
from msehtt.static.form.cochain.vector.static import MseHttStaticCochainVector
from msehtt.tools.vector.static.local import MseHttStaticLocalVector

from msehtt.static.space.num_local_dofs.Lambda.num_local_dofs_m2n2k1 import num_local_dofs__Lambda__m2n2k1_inner
from msehtt.static.space.num_local_dofs.Lambda.num_local_dofs_m2n2k1 import num_local_dofs__Lambda__m2n2k1_outer


class MseHttTimeInstantCochain(Frozen):
    """The cochain instance for ONE particular time!"""

    def __init__(self, f, t):
        """The cochain at time `t`."""
        self._f = f
        self._t = t
        self._gm = self._f.cochain.gathering_matrix
        self._ctype = None
        self._cochain = None
        self._E = None
        self._freeze()

    def __repr__(self):
        super_repr = super().__repr__().split('object')[1]
        return f"<msehtt cochain instant @t={self._t} of form {self._f}" + super_repr

    def _receive(self, cochain):
        """"""
        if isinstance(cochain, dict):
            gm = self._f.cochain.gathering_matrix
            for e in cochain:
                assert e in gm, f"element #{e} is not a rank element."
                assert len(cochain[e]) == gm.num_local_dofs(e), \
                    (f"cochain length wrong for element #{e}. "
                     f"It needs a local cochain of {gm.num_local_dofs(e)} coefficients. "
                     f"Now it gets {len(cochain[e])} coefficients.")
            for e in gm:
                if e not in cochain:
                    assert gm.num_local_dofs(e) == 0, f"For empty dof elements, we can skip the cochain."
                else:
                    pass
            self._ctype = 'dict'
            self._cochain = cochain

        elif callable(cochain):
            self._ctype = 'realtime'
            self._cochain = cochain

        elif cochain.__class__ is MseHttStaticLocalVector:
            assert cochain._gm == self._gm, f"gm must match."
            self._ctype = 'static-local-vector'
            self._cochain = cochain

        else:
            raise NotImplementedError(cochain.__class__.__name__)

    def __getitem__(self, e):
        """"""
        assert e in self._gm, f"element #{e} is not a rank element."

        if self._ctype == 'dict':
            if e in self._cochain:
                return self._cochain[e]
            else:
                return np.array([])

        elif self._ctype == 'realtime':
            return self._cochain(e)

        elif self._ctype == 'static-local-vector':
            return self._cochain[e]

        else:
            raise NotImplementedError()

    def __iter__(self):
        """Go through all rank element indices."""
        for e in self._gm:
            yield e

    def __contains__(self, e):
        """Check whether element indexed ``e`` is a rank element."""
        return e in self._gm

    def components(self, e):
        """Return component wise of cochain in element e."""
        indicator = self._f.space.str_indicator
        element = self._f.tpm.composition[e]
        etype = element.etype
        p = element.degree_parser(self._f.degree)[0]
        if indicator == 'm2n2k1_inner':
            num_local_dofs = num_local_dofs__Lambda__m2n2k1_inner(etype, p)[1]
        elif indicator == 'm2n2k1_outer':
            num_local_dofs = num_local_dofs__Lambda__m2n2k1_outer(etype, p)[1]
        else:
            raise NotImplementedError(f"num_local_dofs for {indicator} space is not implemented.")
        total_local_cochain = self[e]
        start = 0
        local_cochain_components = []
        for num_dofs_component in num_local_dofs:
            end = start + num_dofs_component
            local_cochain_components.append(total_local_cochain[start:end])
            start = end
        return local_cochain_components

    def ___cochain_caller___(self, e):
        """"""
        return self[e]

    @property
    def E(self):
        r"""Incidence matrix."""
        if self._E is None:
            self._E = self._f.space.incidence_matrix(self._f.degree)[0]
        return self._E

    def coboundary(self):
        """exterior derivative; E acting on the cochain; a dict of E(cochain).

        Return a dictionary of the cochain of d(self) at time `t`.
        """
        d_cochain = {}
        for e in self:
            d_cochain[e] = self.E[e] @ self[e]
        return d_cochain

    def ___coboundary_callable___(self, e):
        r""""""
        return self.E[e] @ self[e]

    def of_dof(self, global_dof):
        """Find the cochain for global_dof #global_dof.

        Return the same cochain in all ranks no matter whether the global dof is in an element of the rank.
        """
        elements__local_numbering = self._gm.find_rank_locations_of_global_dofs(global_dof)
        elements__local_numbering = elements__local_numbering[list(elements__local_numbering.keys())[0]]
        cochain = list()
        if len(elements__local_numbering) > 0:
            for location in elements__local_numbering:
                element, local_numbering = location
                cochain.append(
                    self[element][local_numbering]
                )
        cochain = COMM.allgather(cochain)
        COCHAIN = list()
        for _ in cochain:
            COCHAIN.extend(_)
        if len(COCHAIN) == 0:
            raise Exception()
        elif len(COCHAIN) == 1:
            pass
        else:
            for c in COCHAIN[1:]:
                np.testing.assert_almost_equal(c, COCHAIN[0])
        return COCHAIN[0]

    def of_local_dof(self, element_index, local_numbering):
        """Return local cochain for a local dof indicated by element index and local numbering.

        Return the same cochain in all ranks no matter whether the local dof is in the rank.
        """
        if element_index in self:
            cochain = self[element_index][local_numbering]
        else:
            cochain = None
        cochain = COMM.allgather(cochain)
        for ch in cochain:
            if ch is not None:
                return ch
            else:
                pass

    def _merge_to(self, root=MASTER_RANK):
        r""""""
        local_dict = {}
        for e in self:
            local_dict[e] = self[e]
        local_dict = COMM.gather(local_dict, root=root)
        if RANK == root:
            _LOCAL_DICT_ = {}
            for _ in local_dict:
                _LOCAL_DICT_.update(_)
            local_dict = _LOCAL_DICT_
        else:
            pass
        return local_dict

    def __add__(self, other):
        """"""
        if other.__class__ is MseHttStaticCochainVector:
            assert other._gm == self._gm, f"gathering matrices do not match."
            data_dict = {}
            for e in self:
                data_dict[e] = self[e] + other[e]
            return MseHttStaticLocalVector(data_dict, self._gm)

        elif other.__class__ is self.__class__:
            data_dict = {}
            for e in self:
                data_dict[e] = self[e] + other[e]
            return MseHttStaticLocalVector(data_dict, self._gm)

        elif other.__class__ is MseHttStaticLocalVector:
            data_dict = {}
            for e in self:
                data_dict[e] = self[e] + other[e]
            return MseHttStaticLocalVector(data_dict, self._gm)

        else:
            raise NotImplementedError(other.__class__)

    def __sub__(self, other):
        """self - other"""
        if other.__class__ is self.__class__:
            data_dict = {}
            for e in self:
                data_dict[e] = self[e] - other[e]
            return MseHttStaticLocalVector(data_dict, self._gm)
        else:
            raise NotImplementedError()

    def __rmul__(self, other):
        """other * self"""
        if isinstance(other, (int, float)):
            data_dict = {}
            for e in self:
                data_dict[e] = other * self[e]
            return MseHttStaticLocalVector(data_dict, self._gm)
        else:
            raise NotImplementedError()
