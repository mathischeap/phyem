# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from src.form.main import _global_forms


class SpaceFiniteSetting(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._freeze()

    def __repr__(self):
        """customized repr"""
        return f"<SpaceFiniteSetting of {self._space}>"

    @staticmethod
    def specify_form(f, degree):
        """specify a form `f` to be an element of a particular finite dimensional space of degree `degree`."""
        f._degree = degree

    def specify_all(self, degree):
        """Specify all forms of this space to be in the particular finite dimensional space of degree `degree`."""
        for fid in _global_forms:
            f = _global_forms[fid]
            if f.space is self._space and f.is_root() and f._degree is None:
                self.specify_form(f, degree)
