# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from src.wf.mp.linear_system import MatrixProxyLinearSystem
from msehtt.tools.linear_system.dynamic.main import MseHttDynamicLinearSystem

from msehtt.adaptive.mesh.main import MseHtt_Adaptive_TopMesh
from msehtt.adaptive.form.main import MseHtt_Adaptive_TopForm


class MseHtt_Adaptive_Linear_System(Frozen):
    """"""

    def __init__(self, abs_ls):
        """"""
        assert abs_ls.__class__ is MatrixProxyLinearSystem, f"I need a {MatrixProxyLinearSystem}."
        self._abs_ls = abs_ls
        self._msehtt_adaptive_ = None
        self._config = _MSEHTT_ADAPTIVE_LINEAR_SYSTEM_CONFIG_(self)
        self._generation_cache_ = -1
        self._dynamic_cache_ = None
        self._freeze()

    def apply(self):
        r""""""
        generations, BASE = self._msehtt_adaptive_()
        if self._generation_cache_ == generations:
            return self._dynamic_cache_
        else:
            pass

        static_dynamic_LS = MseHttDynamicLinearSystem(
            self._abs_ls, BASE
        ).apply()

        configurations = self.config._configurations
        for CONFIG in configurations:
            args, kwargs = CONFIG
            ARGS = list()
            KWARGS = dict()
            for arg in args:
                if isinstance(arg, tuple) and arg[0] == 'REAL-TIME':
                    ARGS.append(getattr(arg[1], arg[2]))
                else:
                    ARGS.append(arg)

            for key in kwargs:
                value = kwargs[key]
                if isinstance(value, tuple) and value[0] == 'REAL-TIME':
                    KWARGS[key] = getattr(value[1], value[2])
                else:
                    KWARGS[key] = value

            static_dynamic_LS.config(*ARGS, **KWARGS)

        self._generation_cache_ = generations
        self._dynamic_cache_ = static_dynamic_LS

        return static_dynamic_LS

    @property
    def config(self):
        return self._config


class _MSEHTT_ADAPTIVE_LINEAR_SYSTEM_CONFIG_(Frozen):
    r""""""

    def __init__(self, ada_ls):
        r""""""
        self._ada_ls = ada_ls
        self._configurations = list()
        self._freeze()

    def __call__(self, bc_type, *args, **kwargs):
        r""""""
        # ----------- put bc_type info into tuple -----------------------------------------------
        if isinstance(bc_type, list):
            bc_type = tuple(bc_type)
        else:
            assert isinstance(bc_type, tuple), \
                f"pls put bc_type in a list or tuple, indicating (type, category)."
        # ======================================================================================

        if bc_type == ('natural bc', 1):
            configuration = self._conf__natural_bc___1_(*args, **kwargs)
        elif bc_type == ('essential bc', 1):
            configuration = self._config__essential_bc___1_(*args, **kwargs)
        else:
            raise NotImplementedError()

        self._configurations.append(configuration)

    def _conf__natural_bc___1_(self, place, condition, root_form=None):
        r""""""
        if isinstance(place, MseHtt_Adaptive_TopMesh):
            place = ('REAL-TIME', place, 'current')

        else:
            raise NotImplementedError()

        if isinstance(root_form, MseHtt_Adaptive_TopForm):
            root_form = ('REAL-TIME', root_form, 'current')
        else:
            raise NotImplementedError()

        return (('natural bc', 1), place, condition), {'root_form': root_form}

    def _config__essential_bc___1_(self, place, condition, root_form=None):
        r""""""
        if isinstance(place, MseHtt_Adaptive_TopMesh):
            place = ('REAL-TIME', place, 'current')
        else:
            raise NotImplementedError()

        if isinstance(root_form, MseHtt_Adaptive_TopForm):
            root_form = ('REAL-TIME', root_form, 'current')
        else:
            raise NotImplementedError()

        return (('essential bc', 1), place, condition), {'root_form': root_form}
