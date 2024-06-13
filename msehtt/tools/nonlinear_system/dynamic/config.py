# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class MseHtt_Dynamic_NonLinear_System_Config(Frozen):
    """"""

    def __init__(self, d_nls):
        """"""
        self._d_nls = d_nls
        self._configurations = list()
        self._freeze()

    def __call__(self, bc_type, *args, **kwargs):
        """"""
        # ----------- put bc_type info into tuple -----------------------------------------------
        if isinstance(bc_type, list):
            bc_type = tuple(bc_type)
        else:
            assert isinstance(bc_type, tuple), f"pls put bc_type in a list or tuple, indicating (type, category)."
        # ======================================================================================

        if bc_type == ('natural bc', 1):
            # let the linear part take care of the natural BC of type 1.
            self._d_nls._dls.config._conf__natural_bc___1_(*args, **kwargs)
        elif bc_type == ('essential bc', 1):
            self._config__essential_bc___1_(*args, **kwargs)  # pass all inputs to the method, check consistence there.
        else:
            raise NotImplementedError()

    def _config__essential_bc___1_(self, place, condition, root_form):
        """"""
        self._configurations.append(
            {
                'type': 'essential bc',
                'category': 1,
                'place': place,
                'condition': condition,
                'root_form': root_form,
            }
        )
