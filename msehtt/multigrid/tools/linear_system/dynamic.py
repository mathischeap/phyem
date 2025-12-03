# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen
from phyem.src.wf.mp.linear_system import MatrixProxyLinearSystem

from phyem.msehtt.multigrid.main import find_base_on_level
from phyem.msehtt.multigrid.tools.linear_system.static.local import MseHtt_MultiGrid_Static_LocalLinearSystem

from phyem.msehtt.tools.linear_system.dynamic.main import MseHttDynamicLinearSystem


class MseHtt_MultiGrid_DynamicLinearSystem(Frozen):
    r""""""

    def __init__(self, wf_mp_ls, base):
        """"""
        assert wf_mp_ls.__class__ is MatrixProxyLinearSystem, f"I need a {MatrixProxyLinearSystem}."
        self._mp_ls = wf_mp_ls
        self._base = base
        self._levels = {}
        self._configurations = list()
        self._freeze()

    def apply(self):
        r"""The method is used to make the syntax of this MG version to be the same with that of the regular version.

        We do not have to apply in this MG version to continue with the program. But it is ok to do this. At least,
        it checks the program.
        """
        self.get_level().apply()
        return self

    def get_level(self, lvl=None):
        r""""""
        if lvl is None:
            lvl = self._base['the_great_mesh'].max_level
        else:
            pass
        if lvl in self._levels:
            return self._levels[lvl]
        else:
            lvl_base = find_base_on_level(lvl)    # this is a key function.
            lvl_dynamic_linear_system = MseHttDynamicLinearSystem(self._mp_ls, lvl_base)
            lvl_dynamic_linear_system = lvl_dynamic_linear_system.apply()
            self._levels[lvl] = lvl_dynamic_linear_system
            return lvl_dynamic_linear_system

    def pr(self, *args, **kwargs):
        r""""""
        return self.get_level().pr(*args, **kwargs)

    def config(self, bc_type, *args, **kwargs):
        r"""We just save all the configurations."""
        self._configurations.append((bc_type, args, kwargs))

    def __call__(self, *args, **kwargs):
        r"""Return a MultiGrid version of the static local linear system."""
        return MseHtt_MultiGrid_Static_LocalLinearSystem(self, *args, **kwargs)
