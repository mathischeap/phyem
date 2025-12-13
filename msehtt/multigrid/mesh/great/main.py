r"""

"""
from phyem.tools.frozen import Frozen

from phyem.msehtt.static.mesh.great.main import MseHttGreatMesh

from phyem.msehtt.multigrid.default import default_refining_method
from phyem.msehtt.multigrid.default import default_uniform_multigrid_refining_factor
from phyem.msehtt.multigrid.default import default_uniform_max_levels

from phyem.msehtt.multigrid.mesh.great.pass_ import MseHtt_MultiGrid_GreatMesh_Pass
from phyem.msehtt.multigrid.mesh.great.dof_correspondence.main import MseHtt_MultiGrid_GreatMesh_DofCorrespondence
from phyem.msehtt.multigrid.mesh.great.visualize.main import MseHtt_MultiGrid_GreatMesh_Visualize
from phyem.msehtt.multigrid.mesh.great.hierarchy import MseHtt_MultiGrid_GreatMesh_Hierarchy


class MseHtt_MultiGrid_GreatMesh(Frozen):
    r""""""

    def __init__(self):
        r""""""
        self._base_mesh = MseHttGreatMesh()  # the most coarse mesh; the lvl#0 mesh.
        self._msehtt_mesh_args_kwargs = None
        self._configuration = {
            'method': '',             # the method of making multi-levels of grids.
            'parameters': {},         # the parameters for the configuration method
        }
        self._levels = {
            0: self._base_mesh,
        }
        self.___uniform_level_range___ = None
        self._pass = MseHtt_MultiGrid_GreatMesh_Pass(self)
        self._dof_cor_ = MseHtt_MultiGrid_GreatMesh_DofCorrespondence(self)
        self._visualize_ = None
        self._hierarchy_ = MseHtt_MultiGrid_GreatMesh_Hierarchy(self)
        self._freeze()

    def __repr__(self):
        r""""""
        super_repr = super().__repr__().split('object')[1]
        return f"<{self.__class__.__name__}" + super_repr

    def _config(self, *args, mgc=None, **kwargs):
        r"""
        Parameters
        ----------
        *args :
            *args and **kwargs are used to make the base mesh (the most coarse mesh) as a MseHttGreatMesh.
        mgc :
            `multigrid configuration`. To define how we setup the multi-levels of grids based on the
            base mesh (the most coarse mesh), i.e. a MseHttGreatMesh.
        **kwargs :
            *args and **kwargs are used to make the base mesh (the most coarse mesh) as a MseHttGreatMesh.
        """
        if mgc is None:
            mgc = {
                'method': default_refining_method,
                'parameters': {
                    'rff': default_uniform_multigrid_refining_factor,
                    'max-levels': default_uniform_max_levels,
                }
            }
        if isinstance(mgc, int):  # when give a integer to mgc, we use 'uniform' method, and mgc is the max-levels.
            mgc = {
                'method': default_refining_method,
                'parameters': {
                    'rff': default_uniform_multigrid_refining_factor,
                    'max-levels': mgc,
                }
            }
        else:
            pass
        self._msehtt_mesh_args_kwargs = [args, kwargs]
        self._configuration = mgc
        self._base_mesh._config(*args, **kwargs)

    def get_level(self, lvl=None):
        r"""Return the lth level of mesh. level#0 mesh is the base mesh."""
        if lvl is None:
            lvl = self.max_level
        else:
            pass

        if lvl in self._levels:
            return self._levels[lvl]
        else:
            self.___make_mesh_of_level___(lvl)
            return self._levels[lvl]

    def ___make_mesh_of_level___(self, lvl):
        r""""""
        method = self._configuration['method']
        if method == 'uniform':
            self.___make_uniform_mesh_of_level___(lvl)
        else:
            raise NotImplementedError(f"refining using method={method} is not implemented.")

    def ___make_uniform_mesh_of_level___(self, lvl):
        assert isinstance(lvl, int) and 1 <= lvl, \
            (f"[uniform] #{lvl} ({lvl.__class__.__name__}) level is illegal. "
             f"level num must be a integer that >= 1.")
        parameters = self._configuration['parameters']
        rff = parameters['rff']
        assert lvl < parameters['max-levels'], \
            (f"[uniform] Cannot make {lvl}-th level (consider the base mesh as the lvl#0 mesh) "
             f"since max num of levels is {parameters['max-levels']}.")
        lvl_mesh = MseHttGreatMesh()
        args, kwargs = self._msehtt_mesh_args_kwargs
        if rff == 2:
            lvl_mesh._config(*args, **kwargs, ts=lvl, renumbering=True)  # must use renumbering.
        else:
            lvl_mesh._config(*args, **kwargs, ts=lvl, ts_rff=rff, renumbering=True)
            # must use renumbering; Not implemented.
        self._levels[lvl] = lvl_mesh

    @property
    def hierarchy(self):
        """hierarchy of the level meshes."""
        return self._hierarchy_

    # -----------------------------------------------------------------------------------------
    @property
    def max_level(self):
        r""" The most refined level of mesh.
        """
        method_ = self._configuration['method']
        if method_ == 'uniform':
            return self._configuration['parameters']['max-levels'] - 1
        else:
            raise NotImplementedError()

    @property
    def level_range(self):
        r"""Return the range of all possible levels.

        It must be iterable. Its sequence must show a good logic.
        and the max-level mesh must be at its end.

        ANd `level_range[0]` must be 0.
        """
        method_ = self._configuration['method']
        if method_ == 'uniform':
            if self.___uniform_level_range___ is None:
                self.___uniform_level_range___ = range(self._configuration['parameters']['max-levels'])
            return self.___uniform_level_range___
        else:
            raise NotImplementedError()

    def visualize(self, lvl='all', **kwargs):
        r"""The max level visualization."""
        if lvl == 'all':  # plot all levels of this multi-grid mesh properly.
            if self._visualize_ is None:
                self._visualize_ = MseHtt_MultiGrid_GreatMesh_Visualize(self)
            return self._visualize_(**kwargs)
        else:
            return self.get_level(lvl).visualize(**kwargs)

    @property
    def dof_correspondence(self):
        r"""Wrapper of all methods that find the relations between dofs of forms on difference levels."""
        return self._dof_cor_

    @property
    def pass_cochain(self):
        r"""pass a cochain of a form to another form."""
        return self._pass.cochain

    @property
    def pass_vector(self):
        r"""Pass a vector from one level to another level as a cochain."""
        return self._pass.vector_through_cochain

    def find_level(self, obj):
        r"""Find level of the obj."""
        if obj.__class__.__name__ == 'MseHttForm':
            f = obj
            tgm = f.tgm
            for lvl in self._levels:
                if tgm is self._levels[lvl]:
                    return lvl
                else:
                    pass
        else:
            raise NotImplementedError(f"{self.__class__.__name__}: find_level for {obj.__class__.__name__} "
                                      f"is not implemented.")

        raise Exception(f"Find no level, something is wrong?")
