# -*- coding: utf-8 -*-
r"""
"""
from src.config import RANK, MASTER_RANK, COMM
from legacy.generic.py.cochain.time_instant_cochain import ParticularCochainAtTimeInstant


class MPI_PY_Particular_Cochain_At_TimeInstant(ParticularCochainAtTimeInstant):
    """"""
    def __repr__(self):
        """"""
        my_repr = rf"<MPI-PY-Cochain at time={self._t} of "
        rf_repr = self._f.__repr__()
        return my_repr + rf_repr + '>'

    def of_dof(self, i, average=True):
        """The cochain for the global dof `#i`."""
        elements_local_indices = self._f.cochain.gathering_matrix._find_elements_and_local_indices_of_dofs(i)
        i = list(elements_local_indices.keys())[0]
        elements, local_rows = elements_local_indices[i]
        values = list()
        local = self.local
        for e, i in zip(elements, local_rows):
            if e in local:
                values.append(
                    local[e][i]
                )
        values = COMM.gather(values, root=MASTER_RANK)
        if RANK == MASTER_RANK:
            VALUES = list()
            for val in values:
                VALUES.extend(val)

            if average:
                value = sum(VALUES) / len(VALUES)
            else:
                value = VALUES[0]
        else:
            value = None
        return COMM.bcast(value, root=MASTER_RANK)
