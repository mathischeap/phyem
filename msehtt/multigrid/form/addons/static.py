r"""

"""

from phyem.tools.frozen import Frozen


class MseHtt_MultiGrid_FormStaticCopy(Frozen):
    """"""

    def __init__(self, f, t):
        """"""
        self._f = f
        self._t = t
        self._field = None
        self._cochain = None
        self._freeze()

    def __repr__(self):
        super_repr = super().__repr__().split('object')[1]
        return self._f.__repr__().split('at')[0] + f'@ {self._t}' + super_repr

    @property
    def cf(self):
        return self._f.cf[self._t]

    def reduce(self, cf=None):
        r"""Reduce for the msehtt form on all level meshes."""
        for lvl in self._f.tgm.level_range:
            lvl_form_at_t = self._f.get_level(lvl)[self._t]
            if cf is None:
                lvl_form_at_t.cochain = lvl_form_at_t._f.reduce(self.cf)
            else:
                lvl_form_at_t.cochain = lvl_form_at_t._f.reduce(cf)

    @property
    def visualize(self):
        r"""Return a visualizer of the form on the most refined mesh, i.e. the max-level mesh."""
        max_level_form = self._f.get_level()
        return max_level_form.visualize(self._t)

    def error(self, error_type='L2'):
        r"""Return the error of this static form on the max-lvl mesh."""
        max_lvl_form = self._f.get_level()
        max_lvl_form.cf = self._f.cf
        return max_lvl_form[self._t].error(error_type=error_type)

    def norm(self, norm_type='L2', component_wise=False):
        r"""Return the norm of this static form on the max-lvl mesh."""
        max_lvl_form = self._f.get_level()
        return max_lvl_form[self._t].norm(norm_type=norm_type, component_wise=component_wise)

    @property
    def cochain(self):
        r""""""
        if self._cochain is None:
            self._cochain = MseHtt_MG_StaticCopy_Cochain(self)
        return self._cochain


class MseHtt_MG_StaticCopy_Cochain(Frozen):
    r""""""
    def __init__(self, mg_fsc):
        r""""""
        self._mg_fsc = mg_fsc
        self._freeze()

    def of_dof(self, global_dof_number):
        r"""Return a dict that contains all the cochains of dof globally labeled `global_dof_number`
        on all level meshes. The keys of the dict are level indices.

        So be very carefully, may the dof labeled `global_dof_number` is beyond the global dof range
        on a particular level of mesh.
        """
        lvl_range = self._mg_fsc._f.tgm.level_range
        dict_of_cochain_dof = {}
        for lvl in lvl_range:
            lvl_form = self._mg_fsc._f.get_level(lvl)
            lvl_static_form = lvl_form[self._mg_fsc._t]
            lvl_dof_cochain = lvl_static_form.cochain.of_dof(global_dof_number)
            dict_of_cochain_dof[lvl] = lvl_dof_cochain
        return dict_of_cochain_dof

    def of_local_dof(self, element_index, local_numbering):
        r"""Return a dict that contains all the cochains of dof locally labeled `local_numbering` in the element
        `element_index` on all level meshes. The keys of the dict are level indices.

        So be very carefully, may an `element_index` is not included on a particular level of mesh.
        """
        lvl_range = self._mg_fsc._f.tgm.level_range
        dict_of_local_dof_cochain = {}
        for lvl in lvl_range:
            lvl_form = self._mg_fsc._f.get_level(lvl)
            lvl_static_form = lvl_form[self._mg_fsc._t]
            lvl_local_dof_cochain = lvl_static_form.cochain.of_local_dof(element_index, local_numbering)
            dict_of_local_dof_cochain[lvl] = lvl_local_dof_cochain
        return dict_of_local_dof_cochain
