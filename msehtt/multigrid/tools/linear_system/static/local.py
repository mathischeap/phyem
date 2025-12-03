# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen
from phyem.msehtt.multigrid.tools.linear_system.static.global_ import MseHtt_MultiGrid_Static_GlobalLinearSystem


class MseHtt_MultiGrid_Static_LocalLinearSystem(Frozen):
    r"""The static version of the dynamic multigrid linear system (`mg_dls`) called at `*args` and `**kwargs`."""
    def __init__(self, mg_dls, *args, **kwargs):
        r""""""
        self._mg_dls = mg_dls
        self._args = args
        self._kwargs = kwargs
        self._levels = {}
        self._X_of_Axb = None
        self._customize_ = _CUSTOMIZE_(self)
        self._freeze()

    def get_level(self, lvl=None):
        r"""Return the static local linear system on the #lvl-level mesh."""
        if lvl is None:
            lvl = self._mg_dls._base['the_great_mesh'].max_level
        else:
            pass

        if lvl in self._levels:
            return self._levels[lvl]
        else:
            pass

        MG_dynamic_linear_system = self._mg_dls
        lvl_dynamic_linear_system = MG_dynamic_linear_system.get_level(lvl)

        # -------- apply the configurations, IMPORTANT! ------------------------------------------------
        for config in MG_dynamic_linear_system._configurations:
            bc_type, config_arg, config_kwargs = config
            # ----------- put bc_type info into tuple -----------------------------------------------
            if isinstance(bc_type, list):
                bc_type = tuple(bc_type)
            else:
                assert isinstance(bc_type, tuple), f"pls put bc_type in a list or tuple, indicating (type, category)."
            # ======================================================================================
            if bc_type == ('natural bc', 1):
                place, condition = config_arg
                root_form = config_kwargs['root_form']

                lvl_place = place.get_level(lvl)
                lvl_root_form = root_form.get_level(lvl)
                lvl_dynamic_linear_system.config(bc_type, lvl_place, condition, root_form=lvl_root_form)

            elif bc_type == ('essential bc', 1):
                place, condition = config_arg
                root_form = config_kwargs['root_form']

                lvl_place = place.get_level(lvl)
                lvl_root_form = root_form.get_level(lvl)
                lvl_dynamic_linear_system.config(bc_type, lvl_place, condition, root_form=lvl_root_form)

            else:
                raise NotImplementedError(f"bc_type={bc_type} is not implemented. First implement it in "
                                          f"the config class for the MseHtt Dynamic Linear System.")

        lvl_static_local_linear_system = lvl_dynamic_linear_system(*self._args, **self._kwargs)

        # ---------- take care of customizations -----------------------------------------------------
        for cus in self.customize._customizations_:
            indicator = cus[0]
            if indicator == 'set_dof':
                global_dof = cus[1]['global_dof']
                value_dict = cus[1]['value_dict']
                lvl_value = value_dict[lvl]
                lvl_static_local_linear_system.customize.set_dof(global_dof, lvl_value)

            elif indicator == 'set_local_dof':
                ith_unknown = cus[1]['ith_unknown']
                element_index = cus[1]['element_index']
                local_dof_index = cus[1]['local_dof_index']
                value_dict = cus[1]['value_dict']
                lvl_value = value_dict[lvl]
                lvl_static_local_linear_system.customize.set_local_dof(
                    ith_unknown, element_index, local_dof_index, lvl_value
                )

            else:
                raise NotImplementedError(f"indicator={indicator}")

        # ============================================================================================

        self._levels[lvl] = lvl_static_local_linear_system
        return lvl_static_local_linear_system

    def pr(self, *args, **kwargs):
        r""""""
        lvl_static_local_linear_system = self.get_level()
        lvl_static_local_linear_system.pr(*args, **kwargs)

    def assemble(self, **kwargs):
        r""""""
        return MseHtt_MultiGrid_Static_GlobalLinearSystem(self, kwargs)

    @property
    def x(self):
        r"""x (in Ax=b) of this system."""
        if self._X_of_Axb is None:
            self._X_of_Axb = MseHtt_MG_SLS_XXX(self)
        return self._X_of_Axb

    def spy(self, *args, **kwargs):
        r""""""
        return self.get_level().spy(*args, **kwargs)

    @property
    def customize(self):
        return self._customize_


class MseHtt_MG_SLS_XXX(Frozen):
    r"""Class of x (in Ax=b) of the MG static linear system"""
    def __init__(self, s_lls):
        r"""
        Parameters
        ----------
        s_lls :
            multigrid Static Local-Linear-System
        """
        self._s_lls = s_lls
        self._freeze()

    def update(self, x_array):
        r"""This usually is used to make the syntax of MG-version to be the same with the regular version.

        This update normally is already done during solving the MG system.
        """
        lvl_s_lls = self._s_lls.get_level()
        lvl_s_lls.x.update(x_array)


class _CUSTOMIZE_(Frozen):
    r""""""
    def __init__(self, mg_s_llc):
        r""""""
        self._mg_s_llc = mg_s_llc
        self._customizations_ = list()
        self._freeze()

    def set_dof(self, global_dof, value):
        r"""
        Parameters
        ----------
        global_dof :
        value :
            if value is a dict:
                It is a dict whose keys are all level indices and values are the values we want to set the
                dof on the corresponding levels to be.
            elif value is an int or float:
                This means we use the same value for all levels of meshes.
        """
        if isinstance(value, dict):
            value_dict = value
        elif isinstance(value, (int, float)):
            tgm = self._mg_s_llc._mg_dls._base['the_great_mesh']
            value_dict = {}
            for lvl in tgm.level_range:
                value_dict[lvl] = value
        else:
            raise NotImplementedError()

        self._customizations_.append(
            ('set_dof', {'global_dof': global_dof, 'value_dict': value_dict})
        )

    def set_local_dof(self, ith_unknown, element_index, local_dof_index, value):
        r"""
        Parameters
        ----------
        ith_unknown :
        element_index :
        local_dof_index :
        value :
            if value is an int or float:
                This means we use the same value for all levels of meshes.

        """
        if isinstance(value, dict):
            value_dict = value
        elif isinstance(value, (int, float)):
            tgm = self._mg_s_llc._mg_dls._base['the_great_mesh']
            value_dict = {}
            for lvl in tgm.level_range:
                value_dict[lvl] = value
        else:
            raise NotImplementedError()

        self._customizations_.append(
            (
                'set_local_dof',
                {
                     'ith_unknown': ith_unknown,
                     'element_index': element_index,
                     'local_dof_index': local_dof_index,
                     'value_dict': value_dict
                }
            )
        )
