# -*- coding: utf-8 -*-
"""
"""
import numpy as np

from tools.frozen import Frozen
from src.config import COMM


class MseHttTimeInstantCochain(Frozen):
    """"""

    def __init__(self, f, t):
        """"""
        self._f = f
        self._t = t
        self._gm = self._f.cochain.gathering_matrix
        self._ctype = None
        self._cochain = None
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

        else:
            raise NotImplementedError()

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
        else:
            raise NotImplementedError()

    def ___cochain_caller___(self, e):
        """"""
        return self[e]

    def of_dof(self, global_dof):
        """Find the cochain for global_dof #global_dof."""
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
