# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen


class MsePyFormProjection(Frozen):
    """"""

    def __init__(self, from_form):
        """"""
        self._ff = from_form
        self._freeze()

    def __call__(self, to_form, t=None):
        """
        Projection from form `self._ff` to form `to_form`.

        Parameters
        ----------
        to_form
        t :
            We convert the cochain at this time instant.

            If `t` is now, we only convert the cochain at the newest time.

        Returns
        -------

        """
        raise NotImplementedError()
