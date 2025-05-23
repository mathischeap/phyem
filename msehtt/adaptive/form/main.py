# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehtt.static.form.main import MseHttForm
from msehtt.static.mesh.great.main import MseHttGreatMesh
from msehtt.adaptive.form.renew.main import MseHtt_Adaptive_TopForm_CochainRenew
from msehtt.adaptive.form.numeric import MseHtt_Adaptive_TopForm_Numeric

from src.config import RANK, MASTER_RANK, COMM

from msehtt.adaptive.___func___ import ___link_all_forms____
from msehtt.adaptive.___func___ import ___func_renew___
from msehtt.adaptive.___func___ import ___base_tgm___
from msehtt.adaptive.___func___ import ___renew_stamp___


class MseHtt_Adaptive_TopForm(Frozen):
    """"""

    def __init__(self, abstract_form, ___MAX_GENERATIONS____):
        """"""
        self._abstract_form = abstract_form
        self.___MAX_GENERATIONS____ = ___MAX_GENERATIONS____
        self._generations_ = list()
        self._total_generation_ = 0
        self._CoRenew_ = MseHtt_Adaptive_TopForm_CochainRenew(self)
        self.___renew_info___ = None
        self.___msehtt_base___ = None

        self._numeric = MseHtt_Adaptive_TopForm_Numeric(self)

        self._freeze()

    @property
    def ith_generation(self):
        r"""1st generation is the one initialized. And after each renew, this property +=1."""
        return self._total_generation_

    @property
    def abstract(self):
        return self._abstract_form

    @property
    def current(self):
        return self._generations_[-1]

    @property
    def generations(self):
        r"""Return the list of all available generations."""
        return self._generations_

    def ___renew___(self):
        r""""""
        if len(self._generations_) >= self.___MAX_GENERATIONS____:
            self._generations_ = self._generations_[(-self.___MAX_GENERATIONS____+1):]
        else:
            pass
        new_form = MseHttForm(self._abstract_form)
        self._generations_.append(new_form)
        self._total_generation_ += 1

        # ------ renew cf ------------------------------------------------
        if len(self._generations_) > 1:
            old = self._generations_[-2]
            if old.cf._field is None:
                pass
            else:
                self.current.cf = old.cf._field
        else:
            pass

    def ___renew_cochains___(
            self, from_generation=-2, to_generation=-1, renew_cochains='all', use_method=3, clean=False,
    ):
        r""""""
        if len(self._generations_) < 2:
            return None
        else:
            pass
        renew_info = self._CoRenew_(
            from_generation=from_generation, to_generation=to_generation,
            renew_cochains=renew_cochains, use_method=use_method,
            clean=clean,
        )
        return renew_info

    @property
    def cf(self):
        return self.current.cf

    @cf.setter
    def cf(self, _cf):
        r""""""
        if self._total_generation_ == 0:
            raise Exception(f"can only set cf through current form, initialize the form first.")
        else:
            self.current.cf = _cf

    def __getitem__(self, item):
        r""""""
        return self.current[item]

    def __call__(self, t, extrapolate=False):
        """"""
        return self.current(t, extrapolate=extrapolate)

    @property
    def incidence_matrix(self):
        return self.current.incidence_matrix

    def d(self):
        r""""""
        return self.current.d()

    @property
    def name(self):
        r""""""
        return self.current.name

    # ---------------------------------------------------------------------------------------------

    def _make_cache_data(self, t=None):
        r""""""
        current_form_data = self.current._make_cache_data(t=t)

        if RANK == MASTER_RANK:
            renew_info = self.___renew_info___
            return {
                'current_form_data': current_form_data,
                'renew_info': renew_info,
            }

        else:
            return None

    def _read_cache_data(self, cache_data_dict):
        r""""""
        if RANK == MASTER_RANK:
            renew_info = cache_data_dict['renew_info']
            stamp = renew_info['stamp']
        else:
            assert cache_data_dict is None, f"cache data is only read to master rank."
            stamp = None
            renew_info = None

        stamp = COMM.bcast(stamp, root=MASTER_RANK)
        if stamp == self.___msehtt_base___['stamp']:
            pass
        else:
            renew_info = COMM.bcast(renew_info, root=MASTER_RANK)
            self.__update_base___(renew_info)

        # =============================================================
        if RANK == MASTER_RANK:
            time_dict_data = cache_data_dict['current_form_data']
            self.current._read_cache_data(time_dict_data)
        else:
            self.current._read_cache_data(None)

    def __update_base___(self, renew_info):
        trf = renew_info['trf']
        ts = renew_info['ts']
        stamp = renew_info['stamp']

        _base_ = ___base_tgm___(self.___msehtt_base___)
        new_tgm = MseHttGreatMesh()
        new_tgm._config(_base_, trf=trf, ts=ts)

        ___func_renew___(new_tgm, self.___msehtt_base___)

        ___link_all_forms____(new_tgm, self.___msehtt_base___)

        ___renew_stamp___(stamp, trf, ts, self.___msehtt_base___)

    # ------- operators ----------------------------------------------
    def __sub__(self, other):
        r""""""
        if isinstance(other, self.__class__):
            return self.current - other.current
        elif isinstance(other, MseHttForm):
            return self.current - other
        else:
            raise NotImplementedError()

    # =================================================================

    def norm_residual(self):
        r""""""
        return self.current.norm_residual()

    @property
    def numeric(self):
        r""""""
        return self._numeric
