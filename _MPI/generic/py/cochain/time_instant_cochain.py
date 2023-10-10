# -*- coding: utf-8 -*-
r"""
"""
from generic.py.cochain.time_instant_cochain import ParticularCochainAtTimeInstant


class MPI_PY_Particular_Cochain_At_TimeInstant(ParticularCochainAtTimeInstant):
    """"""
    def __repr__(self):
        """"""
        my_repr = rf"<MPI-PY-Cochain at time={self._t} of "
        rf_repr = self._f.__repr__()
        return my_repr + rf_repr + '>'

    def of_dof(self, i, average=True):
        """The cochain for the global dof `#i`."""
        raise NotImplementedError()
