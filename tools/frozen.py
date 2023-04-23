# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
created at: 2/16/2023 12:16 PM
"""

from abc import ABC


class FrozenError(Exception):
    """Raise when we try to define new attribute for a frozen object."""


class Frozen(ABC):
    """Enable a class to freeze self such that no more new attribute can be defined."""

    def __setattr__(self, key, value):
        """"""
        if self.___frozen___ and key not in dir(self):
            raise FrozenError(f" <Frozen> : {self} is frozen. CANNOT define new attributes.")
        object.__setattr__(self, key, value)

    def _freeze(self):
        """Freeze self, can define no more new attributes. """
        self.___FROZEN___ = True

    def _melt(self):
        """Melt self, so  we can define new attributes."""
        self.___FROZEN___ = False

    def _is_frozen(self):
        """Return the status of the form, frozen (True) or melt (False)?"""
        return self.___frozen___

    @property
    def ___frozen___(self):
        """"""
        try:
            return self.___FROZEN___
        except AttributeError:
            object.__setattr__(self, '___FROZEN___', False)
            return self.___FROZEN___
