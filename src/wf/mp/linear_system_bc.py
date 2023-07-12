# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 2:00 PM on 7/5/2023
"""

import sys

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from src.config import _global_operator_lin_repr_setting, _non_root_lin_sep


class _BoundaryCondition(Frozen):
    """"""


class _EssentialBoundaryCondition(_BoundaryCondition):
    """
    provide trace of a form on a boundary section.

    """


class MatrixProxyLinearSystemBoundaryConditions(Frozen):
    """"""

    def __init__(self, ls, mp_bc):
        """"""
        ls = ls._ls

        x, b = ls.x, ls.b

        self._valid_bcs = {}

        for boundary_section_sym_repr in mp_bc:

            bcs = mp_bc[boundary_section_sym_repr]

            if len(bcs) == 0:  # section defined, but no valid boundary condition imposed on it.
                pass

            else:  # there are valid boundary conditions on this boundary section.

                self._valid_bcs[boundary_section_sym_repr] = list()

                for raw_bc in bcs:

                    bc_pattern = self._parse_bc_pattern(raw_bc)

                    self._valid_bcs[boundary_section_sym_repr].append(
                        bc_pattern
                    )

        self._freeze()

    def _parse_bc_pattern(self, raw_bc):
        """We study the raw bc item and retrieve the correct BoundaryCondition object here!

        If some boundary condition type is not recognized, raise Error here!
        """
        print(raw_bc)

        return None


    def __iter__(self):
        """go through all boundary section sym_repr that has valid BC on."""
        for boundary_sym_repr in self._valid_bcs:
            yield boundary_sym_repr

    def __getitem__(self, boundary_section_sym_repr):
        """Return the B.Cs on this boundary section."""
        assert boundary_section_sym_repr in self, f"no valid BC is defined on {boundary_section_sym_repr}."
        return self._valid_bcs[boundary_section_sym_repr]


if __name__ == '__main__':
    # python 
    pass
