
from phyem.tools.frozen import Frozen
from phyem.msehtt.tools.nonlinear_system.static_composite.solve.main import (
    MseHtt_Static_Composite_Nonlinear_System_Solve)


class MseHtt_Static_Composite_NonLinear_System(Frozen):
    """Must be local-wise (rank-wise)."""

    def __init__(self, *static_nonlinear_systems, condlist=None):
        r"""
        Parameters
        ----------
        static_nonlinear_systems :
            The static nonlinear systems to be used.
        conditions :
            The conditions that determine in each element which nonlinear system should be selected.
        """
        self._static_nonlinear_systems = static_nonlinear_systems
        self._condlist_ = condlist

        self._global_row_gathering_matrix = static_nonlinear_systems[0]._global_row_gm
        self._global_col_gathering_matrix = static_nonlinear_systems[0]._global_col_gm_

        for k, sns in enumerate(static_nonlinear_systems):
            assert sns._global_row_gm == self._global_row_gathering_matrix, f"{k}th global row gm does not match"
            assert sns._global_col_gm_ == self._global_col_gathering_matrix, f"{k}th global col gm does not match"

        self._system_usage_dict_ = None
        self._solve_ = MseHtt_Static_Composite_Nonlinear_System_Solve(self)
        self._freeze()

    @property
    def condlist(self):
        if self._condlist_ is None:
            raise Exception(f"set condlist first.")
        return self._condlist_

    @property
    def system_usage_dict(self):
        if self._system_usage_dict_ is None:
            system_usage_dict = dict()
            # system_usage_dict[i] = k means in local element #i,
            # we use local system of {k}th static local linear system
            conditions = self.condlist
            for i in self._global_row_gathering_matrix:  # go through all local element indices
                for k, cond in enumerate(conditions):
                    ToF = cond(i)
                    if ToF:
                        system_usage_dict[i] = k
                        break
                    else:
                        pass
                assert i in system_usage_dict, f"Cannot find a local system for element #{i}. Check the condition list."
            self._system_usage_dict_ = system_usage_dict

        return self._system_usage_dict_

    def visualize(self):
        r"""Visualize the composition."""
        system_usage_dict = self.system_usage_dict
        nonlinear_system = self._static_nonlinear_systems[0]
        unknown = nonlinear_system.unknowns[0]
        tgm = unknown.tgm
        tgm.visualize(distribution=system_usage_dict)

    @property
    def solve(self):
        return self._solve_
