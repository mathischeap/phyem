# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehy.tools.nonlinear_operator.static.local import IrregularStaticLocalNonlinearOperator


class IrregularDynamicLocalNonlinearOperator(Frozen):
    """"""

    def __init__(self, data, *correspondence, direct_derivative_contribution=False):
        """"""
        if callable(data):
            # we receive a dictionary of n-d array
            self._dtype = 'static'
            self._data = data
        else:
            raise NotImplementedError(f"MsePyDynamicLocalMatrix cannot take {data}.")

        for form in correspondence:
            assert hasattr(form, '_is_discrete_form') and form._is_discrete_form(), \
                f"corresponding forms must be a discrete form."

        self._correspondence = correspondence
        assert isinstance(direct_derivative_contribution, bool), \
            f"direct_derivative_contribution must be bool."
        self._direct_derivative_contribution = direct_derivative_contribution
        self._ndim = len(correspondence)
        self._tf = None
        self._freeze()

    def __call__(self, *args, g=None, **kwargs):
        """Gives a static local matrix by evaluating the dynamic local matrix with `*args, **kwargs`."""
        # first we find the particular forms for the static object
        particular_forms = list()
        times = self._time_caller(*args, g=g, **kwargs)
        generations = self._generation_caller(*args, g=g, **kwargs)
        for i, generic_form in enumerate(self._correspondence):
            if generic_form is self.test_form:
                particular_forms.append(
                    generic_form   # the test-form, it is not at any time instant
                )
            else:
                particular_forms.append(
                    generic_form[(times[i], generations[i])]  # a form copy at a particular time instant.
                )

        for g in generations:
            assert g == generations[0], f"generations={generations} is wrong, all generation must be the same."
        g = generations[0]

        # make the static msepy mda.
        if self._dtype == 'static':
            static = IrregularStaticLocalNonlinearOperator(
                self._data(g),
                g,
                particular_forms,
                direct_derivative_contribution=self._direct_derivative_contribution,
                cell_range=None,  # since we use a dict of array. `cell range` will be the dict keys.
            )
        else:
            raise NotImplementedError(f"data type = {self._dtype} is wrong!")

        # check and return
        assert isinstance(static, IrregularStaticLocalNonlinearOperator), f"must return a static one!"
        return static

    def _time_caller(self, *args, g=None, **kwargs):
        _ = g
        times = list()
        for generic_form in self._correspondence:
            if generic_form is self.test_form:
                times.append(None)
            else:
                times.append(
                    generic_form.cochain._ati_time_caller(*args, g=g, **kwargs)
                )
        return times

    def _generation_caller(self, *args, g=None, **kwargs):
        """"""
        _ = args, kwargs
        generations = list()
        for generic_form in self._correspondence:
            generations.append(
                generic_form.cochain._generation_caller(*args, g=g, **kwargs)
            )
        return generations

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
