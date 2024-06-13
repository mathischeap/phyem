# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from typing import Dict
from src.form.main import Form
from msehtt.tools.matrix.static.local import MseHttStaticLocalMatrix
from msehtt.static.form.cf import MseHttStaticFormCF
from msehtt.static.form.addons.static import MseHttFormStaticCopy
from msehtt.static.form.addons.ic import MseHtt_From_InterpolateCopy
from msehtt.static.form.cochain.main import MseHttCochain
from msehtt.static.form.visualize.main import MseHttFormVisualize
from msehtt.static.form.bi.main import MseHttStaticForm_Boundary_Integrate
from msehtt.static.form.numeric.main import MseHtt_Form_Numeric


class MseHttForm(Frozen):
    """"""

    def __init__(self, abstract_root_form):
        """"""
        self._pAti_form: Dict = {
            'base_form': None,  # the base form
            'ats': None,  # abstract time sequence
            'ati': None,  # abstract time instant
        }
        self._ats_particular_forms = dict()   # the abstract forms based on this form.
        assert abstract_root_form.__class__ is Form, f"I need an abstract form."
        self._degree = abstract_root_form._degree
        self._abstract = abstract_root_form

        self._tgm = None  # the msehtt great mesh
        self._tpm = None  # the msehtt partial mesh
        self._space = None  # the msehtt space
        self._manifold = None   # the msehtt manifold

        self._cf = None
        self._cochain = None
        self._im = None
        self._bi = None
        self._numeric = None
        self._export = None
        self._project = None
        self._freeze()

    @property
    def abstract(self):
        """Return the abstract form."""
        return self._abstract

    def __repr__(self):
        """repr"""
        ab_rf_repr = self._abstract.__repr__().split(' at ')[0][1:]
        return "<MseHtt " + ab_rf_repr + super().__repr__().split(" object")[1]

    @property
    def degree(self):
        """The degree of the form."""
        return self._degree

    def _is_base(self):
        """Am I a base root-form (not abstracted at a time)?"""
        return self._base is None

    @property
    def _base(self):
        """The base root-form I have.

        if `self._is_base()`, return None. Else return the base form.
        """
        return self._pAti_form['base_form']

    @property
    def tgm(self):
        """Return the msehtt great mesh."""
        return self._tgm

    @property
    def tpm(self):
        """Return the msehtt partial mesh."""
        return self._tpm

    @property
    def space(self):
        """Return the msehtt space."""
        return self._space

    @property
    def manifold(self):
        """Return the msehtt manifold."""
        return self._manifold

    @property
    def cf(self):
        """Continuous form."""
        if self._is_base():
            if self._cf is None:
                self._cf = MseHttStaticFormCF(self)
            return self._cf
        else:
            return self._base.cf

    @cf.setter
    def cf(self, _cf):
        """"""
        if self._is_base():
            self.cf.field = _cf
        else:
            self._base.cf = _cf

    def __getitem__(self, t):
        """"""
        if t is None:
            t = self.cochain.newest  # newest time
        elif isinstance(t, str):
            # when use str, we are looking for the form at a time step.
            from src.time_sequence import _global_abstract_time_sequence
            if len(_global_abstract_time_sequence) == 1:
                ts_indicator = list(_global_abstract_time_sequence.keys())[0]
                ts = _global_abstract_time_sequence[ts_indicator]
                abstract_time_instant = ts[t]
                t = abstract_time_instant()()
            else:
                raise Exception(f"multiple time sequences exist. "
                                f"I don't know which one you are referring to")
        else:
            pass
        t = self.cochain._parse_t(t)  # round off the truncation error to make it clear.

        if isinstance(t, (int, float)):
            if self._is_base():
                return MseHttFormStaticCopy(self, t)
            else:
                return MseHttFormStaticCopy(self._base, t)
        else:
            raise Exception(f"cannot accept t={t}.")

    def __call__(self, t, extrapolate=False):
        """"""
        if isinstance(t, str):
            # when use str, we are looking for the form at a time step.
            from src.time_sequence import _global_abstract_time_sequence
            if len(_global_abstract_time_sequence) == 1:
                ts_indicator = list(_global_abstract_time_sequence.keys())[0]
                ts = _global_abstract_time_sequence[ts_indicator]
                abstract_time_instant = ts[t]
                t = abstract_time_instant()()
            else:
                raise Exception(f"multiple time sequences exist. "
                                f"I don't know which one you are referring to")
        else:
            pass

        t = self.cochain._parse_t(t)

        if isinstance(t, (int, float)):
            if self._is_base():
                return MseHtt_From_InterpolateCopy(self, t, extrapolate=extrapolate)
            else:
                return MseHtt_From_InterpolateCopy(self._base, t, extrapolate=extrapolate)
        else:
            raise Exception(f"cannot accept t={t}.")

    @property
    def cochain(self):
        """The cochain class."""
        if self._cochain is None:
            self._cochain = MseHttCochain(self)
        return self._cochain

    def reduce(self, cf_at_t):
        """"""
        return self._space.reduce(cf_at_t, self.degree)

    def reconstruct(self, cochain, *meshgrid, ravel=False):
        """"""
        return self._space.reconstruct(self.degree, cochain, *meshgrid, ravel=ravel)

    def error(self, cf, cochain, error_type='L2'):
        """"""
        return self._space.error(cf, cochain, self.degree, error_type=error_type)

    def visualize(self, t):
        if self._is_base():
            return MseHttFormVisualize(self, t)
        else:
            return self._base.visualize(t)

    @property
    def incidence_matrix(self):
        if self._im is None:
            gm0 = self.space.gathering_matrix._next(self.degree)
            gm1 = self.cochain.gathering_matrix
            E, cache_key_dict = self.space.incidence_matrix(self.degree)
            self._im = MseHttStaticLocalMatrix(
                E,
                gm0,
                gm1,
                cache_key=cache_key_dict
            )
        return self._im

    def norm(self, cochain, norm_type='L2'):
        """"""
        return self.space.norm(self.degree, cochain, norm_type=norm_type)

    @property
    def bi(self):
        """boundary integrate."""
        if self._bi is None:
            self._bi = MseHttStaticForm_Boundary_Integrate(self)
        return self._bi

    def reconstruction_matrix(self, *meshgrid):
        """Return the reconstruction matrix for all rank elements."""
        return self.space.reconstruction_matrix(self.degree, *meshgrid)

    @property
    def numeric(self):
        """"""
        if self._is_base():
            if self._numeric is None:
                self._numeric = MseHtt_Form_Numeric(self)
            return self._numeric
        else:
            return self._base.numeric

    def norm_residual(self, from_time=None, to_time=None, norm_type='L2'):
        """By default, use L2-norm."""
        if to_time is None or from_time is None:
            all_cochain_times = list(self.cochain._tcd.keys())
            all_cochain_times.sort()

            if to_time is None:
                if len(all_cochain_times) == 0:
                    to_time = None
                else:
                    to_time = all_cochain_times[-1]
            else:
                pass

            if from_time is None:
                if len(all_cochain_times) == 0:
                    from_time = None
                elif len(all_cochain_times) == 1:
                    from_time = all_cochain_times[-1]
                else:
                    from_time = all_cochain_times[-2]
            else:
                pass

        else:
            pass

        if from_time is None:
            assert to_time is None, f"cochain must be of no cochain."
            return np.nan  # we return np.nan.
        else:
            assert isinstance(from_time, (int, float)), f"from_time = {from_time} is wrong."
            assert isinstance(to_time, (int, float)), f"to_time = {to_time} is wrong."
            from_time = self.cochain._parse_t(from_time)
            to_time = self.cochain._parse_t(to_time)

            if from_time == to_time:
                return np.nan  # return nan since it is not a residual

            else:
                from_cochain = self.cochain[from_time]
                to_cochain = self.cochain[to_time]
                diff_cochain = to_cochain - from_cochain
                norm = self.space.norm(self.degree, diff_cochain, norm_type=norm_type)
                return norm
