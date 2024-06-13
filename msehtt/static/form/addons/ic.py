# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class MseHtt_From_InterpolateCopy(Frozen):
    """"""

    def __init__(self, f, t, extrapolate=False):
        """"""
        self._f = f
        self._t = t
        self._cochain = self._find_cochain_at_arbitrary_time_instant(
            f, t, extrapolate=extrapolate
        )
        self._freeze()

    def _find_cochain_at_arbitrary_time_instant(self, f, t, extrapolate=False):
        """"""
        exact_cochain_times = list(f.cochain._tcd.keys())
        exact_cochain_times.sort()
        if len(exact_cochain_times) <= 1:
            raise Exception(f"not enough cochain to interpolate.")
        else:
            if t < exact_cochain_times[0]:
                assert extrapolate, f"t < lower bound time, must turn on extrapolate."
                return self._extrapolate_(f, t)
            elif t > exact_cochain_times[-1]:
                assert extrapolate, f"t > upper bound time, must turn on extrapolate."
                return self._extrapolate_(f, t)
            else:
                lower_bound, upper_bound = 0, 0
                for i, lower_bound in enumerate(exact_cochain_times):
                    upper_bound = exact_cochain_times[i+1]
                    if lower_bound <= t <= upper_bound:
                        break
                    else:
                        pass
                assert lower_bound != upper_bound, f"must be!"
                delta_lower = t - lower_bound
                total_delta = upper_bound - lower_bound
                delta = delta_lower / total_delta
                lower_cochain = f.cochain[lower_bound]
                upper_cochain = f.cochain[upper_bound]
                cochain = lower_cochain + delta * (upper_cochain - lower_cochain)
                return cochain

    def _extrapolate_(self, f, t):
        """"""
        raise NotImplementedError()

    @property
    def cochain(self):
        return self._cochain
