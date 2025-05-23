# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehtt.adaptive.form.renew.method_1_global_interpolation import MseHtt_FormCochainRenew_Method1_GlobalInterpolation
from msehtt.adaptive.form.renew.method_2_base_element import MseHtt_FormCochainRenew_Method2_BaseElementWise
from msehtt.adaptive.form.renew.method_3_local_interpolation import MseHtt_FormCochainRenew_Method3_LocalInterpolation


class MseHtt_Adaptive_TopForm_CochainRenew(Frozen):
    r""""""

    def __init__(self, tpf):
        r""""""
        self._tpf = tpf
        self._freeze()

    def __call__(
            self, from_generation=-2, to_generation=-1,
            renew_cochains=0, use_method=3, clean=False, **kwargs,
    ):
        r"""

        Parameters
        ----------
        from_generation
        to_generation
        renew_cochains :
            Which cochains (indicated by cochain time) of `from_generation` to be renewed to `to_generation`.

            If `renew_cochains` == 'all', all cochains will be renewed to the target generation.
        use_method :
            Which method to be used for this renew?

        Returns
        -------

        """
        f_form = self._tpf.generations[from_generation]
        t_form = self._tpf.generations[to_generation]

        # -------- check which cochains to be renewed to the dest generation ----------------------------
        if isinstance(renew_cochains, str) and renew_cochains == 'all':
            # renew all cochains
            renew_cochains = 'all'
        else:
            raise NotImplementedError()

        # -------- collect cochains to be renewed -------------------------------------------------------
        available_cochain_times = f_form.cochain.times

        if isinstance(renew_cochains, str) and renew_cochains == 'all':
            cochain_times_tobe_renewed = available_cochain_times
        else:
            raise NotImplementedError()

        # --------- decide renew or pass -----------------------------------------------------------------
        if len(cochain_times_tobe_renewed) == 0:
            return {}
        else:
            # SELECT THE METHOD
            if use_method == 1:
                renewer = MseHtt_FormCochainRenew_Method1_GlobalInterpolation(f_form, t_form)
            elif use_method == 2:
                renewer = MseHtt_FormCochainRenew_Method2_BaseElementWise(f_form, t_form)
            elif use_method == 3:
                renewer = MseHtt_FormCochainRenew_Method3_LocalInterpolation(f_form, t_form)
            else:
                raise NotImplementedError()

            # DO THE RENEWING
            renew_info = renewer(cochain_times_tobe_renewed, clean=clean, **kwargs)

            return renew_info  # since changes are made locally in the cochains.
