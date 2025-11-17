# -*- coding: utf-8 -*-
r"""
"""

from phyem.msehtt.static.form.addons.nop_data_computer.trilinear_AxB_ip_C import AxB_ip_C

_cache_AxB_dp_C_3d_data_ = {}


class AxB_dp_C(AxB_ip_C):
    """"""

    def _make_3d_data(self):
        """"""
        # ---- if the data is already there -------------------------------------------
        if self._3d_data is not None:
            pass
        # ---- if the data is cached ---------------------------------------------------
        elif self._cache_key in _cache_AxB_dp_C_3d_data_:
            self._3d_data = _cache_AxB_dp_C_3d_data_[self._cache_key]
        # ------- make the data -------------------------------------------------------------
        else:
            _3d_data = self._generate_data_()
            self._3d_data = _3d_data
            _cache_AxB_dp_C_3d_data_[self._cache_key] = _3d_data

    @classmethod
    def clean_cache(cls):
        r""""""
        keys = list(_cache_AxB_dp_C_3d_data_.keys())
        for key in keys:
            del _cache_AxB_dp_C_3d_data_[key]
