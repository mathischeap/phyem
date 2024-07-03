# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from msehtt.tools.nonlinear_system.operator.local_multilinear import MseHttStatic_Local_Multi_Linear_NonlinearOperator
from msehtt.tools.nonlinear_system.operator.local_wrapper import MseHttStatic_Local_Wrapper_NonlinearOperator


class MseHttDynamicLocalNonlinearOperator(Frozen):
    """"""

    def __init__(self, data, *correspondence, multi_linear=False):
        """"""
        if data.__class__ is dict:
            for i in data:
                assert data[i].__class__ is np.ndarray, f"when providing a dict, it must be a dict of nd array."
            # we receive a dictionary of n-d array
            self._dtype = 'static'
            self._data = data
        else:
            raise NotImplementedError(f"MsePyDynamicLocalMatrix cannot take {data}.")

        self._correspondence = correspondence
        assert isinstance(multi_linear, bool), f"multi_linear must be bool."
        self._multi_linear = multi_linear
        self._ndim = len(correspondence)
        self._tf = None
        self._freeze()

    def __call__(self, *args, **kwargs):
        """Gives a static local matrix by evaluating the dynamic local matrix with `*args, **kwargs`."""
        # first we find the particular forms for the static object
        particular_forms = list()
        times = self._time_caller(*args, **kwargs)
        for i, generic_form in enumerate(self._correspondence):
            if generic_form is self.test_form:
                particular_forms.append(
                    generic_form   # the test-form, it is not at any time instant
                )
            else:
                particular_forms.append(
                    generic_form[times[i]]  # a form copy at a particular time instant.
                )

        # make the static msepy mda.
        if self._dtype == 'static':
            assert self._multi_linear is True, f"static nonlinear term must be multi-linear."
            static = MseHttStatic_Local_Multi_Linear_NonlinearOperator(
                self._data,
                particular_forms,
                direct_derivative_contribution=self._multi_linear,
            )
        elif self._dtype == 'wrapper':
            static = MseHttStatic_Local_Wrapper_NonlinearOperator(
                self._data,
                particular_forms,
            )
        else:
            raise NotImplementedError(f"data type = {self._dtype} is wrong!")

        # check and return
        assert isinstance(static, MseHttStatic_Local_Multi_Linear_NonlinearOperator), f"must return a static one!"
        return static

    def _time_caller(self, *args, **kwargs):
        times = list()
        for generic_form in self._correspondence:
            if generic_form is self.test_form:
                times.append(None)
            else:
                times.append(
                    generic_form.cochain._ati_time_caller(*args, **kwargs)
                )
        return times

    @property
    def correspondence(self):
        """The correspondence of multi-dimensions."""
        return self._correspondence

    @property
    def ndim(self):
        """the dimensions of the MDA."""
        return self._ndim

    @property
    def test_form(self):
        """"""
        return self._tf

    @test_form.setter
    def test_form(self, tf):
        """"""
        assert tf in self._correspondence, f"tf is not in the correspondence."
        self._tf = tf
