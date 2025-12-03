r"""
"""
from phyem.tools.frozen import Frozen
from phyem.msehtt.multigrid.tools.linear_system.static.solvers.direct import (
    _msehtt_multigrid_direct_linear_system_solver_)


class MseHtt_MultiGrid_Static_GlobalLinearSystem(Frozen):
    r"""The static version of the dynamic multigrid linear system (`mg_dls`) called at `*args` and `**kwargs`."""
    def __init__(self, static_local_linear_system, assemble_kwargs):
        r""""""
        self._s_l_ls_ = static_local_linear_system
        self._assemble_kwargs_ = assemble_kwargs
        self._freeze()

    def get_level(self, lvl=None):
        if lvl is None:
            lvl = self._s_l_ls_._mg_dls._base['the_great_mesh'].max_level
        else:
            pass

        lvl_static_local_linear_system = self._s_l_ls_.get_level(lvl)
        lvl_assemble_kwargs = {}
        for key in self._assemble_kwargs_:
            if key == 'cache':
                base_cache_key = self._assemble_kwargs_['cache']
                if base_cache_key is None:
                    pass
                else:
                    lvl_assemble_kwargs['cache'] = base_cache_key + f'@{lvl}'
            else:
                lvl_assemble_kwargs[key] = self._assemble_kwargs_[key]
        return lvl_static_local_linear_system.assemble(**lvl_assemble_kwargs)

    def solve(self, scheme, x0=None, **kwargs):
        r""""""
        if scheme in ('direct', 'spsolve'):  # we use direct solver on the coarsest lvl to solve the system.
            # Then the result is passed to the next level of mesh and we solve it with lgmres
            # whose kwargs is in `kwargs`.
            _ = x0  # we will not use x0
            return _msehtt_multigrid_direct_linear_system_solver_(self, kwargs)
        else:
            raise NotImplementedError(f"{self.__class__.__name__}'s solve scheme {scheme} is not implemented.")

    def spy(self, *args, **kwargs):
        r"""spy the assembled system on the max-level mesh."""
        return self.get_level().spy(*args, **kwargs)

    @property
    def condition_number(self):
        return self.get_level().condition_number

    @property
    def rank(self):
        return self.get_level().rank

    @property
    def num_singularities(self):
        return self.get_level().num_singularities

    @property
    def shape(self):
        return self.get_level().shape
