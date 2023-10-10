# -*- coding: utf-8 -*-
r"""
"""
from generic.py.cochain.main import Cochain, CochainLockedError
from _MPI.generic.py.cochain.time_instant_cochain import MPI_PY_Particular_Cochain_At_TimeInstant


class MPI_PY_Form_Cochain(Cochain):
    """"""

    def _set(self, t, cochain):
        """add to cochain at `t` to be cochain."""
        if self._locker:  # cochain locked, cannot set new cochain.
            raise CochainLockedError(f"Cochain of {self._f} is locked!")
        else:
            pass

        rf = self._f

        if rf._is_base():
            t = self._parse_t(t)

            _cochain_at_time = MPI_PY_Particular_Cochain_At_TimeInstant(self._f, t)

            _cochain_at_time._receive(cochain)

            self._tcd[t] = _cochain_at_time
            self._newest_t = t

        else:
            rf._base.cochain._set(t, cochain)

    # def static_vec(self, t):
    #     """"""
    #     assert isinstance(t, (int, float)), f"t={t} is wrong."
    #     if t in self:
    #         return MPI_PY_Localize_Static_Vector_Cochain(self._f, t, self[t].local, self.gathering_matrix)
    #     else:
    #         # this one is usually used to receive a cochain.
    #         return MPI_PY_Localize_Static_Vector_Cochain(self._f, t, None, self.gathering_matrix)
    #         # the data is None (empty)
    #
    # @property
    # def dynamic_vec(self):
    #     """"""
    #     return MPI_PY_Localize_Dynamic_Vector_Cochain(self._dynamic_cochain_caller)
