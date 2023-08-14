# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from msepy.tools.multidimensional_array.static.local import MsePyStaticLocalMDA
from msepy.form.main import MsePyRootForm


class MsePyDynamicLocalMDA(Frozen):
    """"""

    def __init__(self, data, *correspondence, modes=None):
        """"""
        if data.__class__ is dict:
            for i in data:
                assert data[i].__class__ is np.ndarray, f"when providing a dict, it must be a dict of nd array."
            # we receive a dictionary of n-d array
            self._dtype = 'static'
            self._data = data
        else:
            raise NotImplementedError(f"MsePyDynamicLocalMatrix cannot take {data}.")

        for form in correspondence:
            assert form.__class__ is MsePyRootForm, f"corresponding forms must be {MsePyRootForm}."

        if modes is None:
            modes = 'homogeneous'
        else:
            pass

        # modes will affect the way of computing derivatives.
        assert modes in (
            'homogeneous',  # different axes represent different variables, and connected by only multiplication.
            # for example, a * b * c.
        )

        self._modes = modes
        self._correspondence = correspondence
        self._ndim = len(correspondence)
        self._tf = None
        self._freeze()

    def __call__(self, *args, **kwargs):
        """Gives a static local matrix by evaluating the dynamic local matrix with `*args, **kwargs`."""
        if self._dtype == 'static':
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
            static = MsePyStaticLocalMDA(self._data, particular_forms, modes=self._modes)
        else:
            raise NotImplementedError(f"data type = {self._dtype} is wrong!")

        assert isinstance(static, MsePyStaticLocalMDA), f"must return a static one!"

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
