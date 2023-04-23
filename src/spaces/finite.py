# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
created at: 3/16/2023 5:29 PM
"""
import sys

if './' not in sys.path:
    sys.path.append('./')

from tools.frozen import Frozen
from src.form.main import _global_forms


class SpaceFiniteSetting(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._degrees_form_dict = dict()  # keys are degrees of the particular finite dimensional spaces,
        # and values are forms in these particular fintie dimensional spaces.
        self._all_finite_forms = set()
        self._freeze()

    def __repr__(self):
        """customized repr"""
        return f"<SpaceFiniteSetting of {self._space}>"

    def __len__(self):
        """how many particular finite dimensional spaces?"""
        return len(self._degrees_form_dict)

    def __iter__(self):
        """Go through all degrees of particular finite dimensional spaces"""
        for degree in self._degrees_form_dict:
            yield degree

    def __contains__(self, degree):
        """if there is a particular finite dimensional space of a certain degree?"""
        return degree in self._degrees_form_dict

    def __getitem__(self, degree):
        """return the forms in the particular finite dimensional space of this a certain degree"""
        assert degree in self._degrees_form_dict, f"I have no finite dimensional space of degree {degree}."
        return self._degrees_form_dict[degree]

    def new(self, degree):
        """define a new finite dimensional space of `degree`.

        We must define new degree through this method.
        """
        assert isinstance(degree, (int, float, list, tuple)), \
            f"Can only use int, float, list or tuple for the degree."
        if isinstance(degree, list):
            degree = tuple(degree)
        else:
            pass
        if isinstance(degree, tuple):
            for i, d in enumerate(degree):
                assert isinstance(d, (int, float)), \
                    f"degree[{i}] = {d} is not valid, must be a int or integer."

        if degree in self:
            pass
        else:
            self._degrees_form_dict[degree] = list()

        return degree

    def specify_form(self, f, degree):
        """specify a form `f` to be an element of a particular finite dimensional space of degree `degree`."""
        assert f not in self._all_finite_forms, f"form {f} is already in."
        degree = self.new(degree)  # must do this! We will parse the `degree` here!
        self[degree].append(f)
        self._all_finite_forms.add(f)
        f._degree = degree

    def specify_all(self, degree):
        """Specify all forms of this space to be in the particular finite dimensional space of degree `degree`."""
        for fid in _global_forms:
            f = _global_forms[fid]
            if f.space is self._space and f.is_root() and f._degree is None:
                self.specify_form(f, degree)


if __name__ == '__main__':
    # python src/spaces/finite.py
    import __init__ as ph

    m = ph.manifold(3)
    m = ph.mesh(m)

    O0 = ph.space.new('Omega', 0)

    finite = O0.finite
    finite.new(3)
    finite.new(4)
    finite.new([1, 2, 3])
