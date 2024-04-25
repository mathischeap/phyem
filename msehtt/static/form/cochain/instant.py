# -*- coding: utf-8 -*-
"""
"""
import numpy as np

from tools.frozen import Frozen


class MseHttTimeInstantCochain(Frozen):
    """"""

    def __init__(self, f, t):
        """"""
        self._f = f
        self._t = t
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
        assert e in self._f.cochain.gathering_matrix, f"element #{e} is not a rank element."
        if self._ctype == 'dict':
            if e in self._cochain:
                return self._cochain[e]
            else:
                return np.array([])

        elif self._ctype == 'realtime':
            return self._cochain(e)
        else:
            raise NotImplementedError()
