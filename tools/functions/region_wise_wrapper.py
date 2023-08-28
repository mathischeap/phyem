# -*- coding: utf-8 -*-
r"""
We use this wrapper to wrap a series of functions in a region (dict keys) -wise structure. We en-rich the
dict class such that we can define more operations to the functions.
"""


class RegionWiseFunctionWrapper(dict):
    """Use this wrapper to enable more operations."""

    def __neg__(self):
        new_neg_field = RegionWiseFunctionWrapper()
        for i in self:
            new_neg_field[i] = - self[i]
        return new_neg_field

    def __sub__(self, other):
        """"""
        if other.__class__ is self.__class__:
            assert len(other) == len(self), f"region number differs."

            new_dict = dict()

            for region in self:
                new_dict[region] = self[region] - other[region]

            return self.__class__(new_dict)

        else:
            raise NotImplementedError()

    def __add__(self, other):
        """"""
        if other.__class__ is self.__class__:
            assert len(other) == len(self), f"region number differs."

            new_dict = dict()

            for region in self:
                new_dict[region] = self[region] + other[region]

            return self.__class__(new_dict)

        else:
            raise NotImplementedError()
