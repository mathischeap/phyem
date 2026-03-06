r"""

"""

from phyem.tools.frozen import Frozen


class MseHttLocalScalar(Frozen):
    r""""""

    def __init__(self, data):
        r""""""
        self._receive_data(data)
        self._freeze()

    def _receive_data(self, data):
        r""""""
        assert isinstance(data, dict)
        self._data_ = data

    def _get_meta_data_(self, element_index):
        r""""""
        return self._data_[element_index]

    def __getitem__(self, element_index):
        r""""""
        return self._get_meta_data_(element_index)

    def __iter__(self):
        r""""""
        for e in self._data_:
            yield e

    def __add__(self, other):
        r""""""
        if other.__class__ is self.__class__:
            new_data_dict = {}
            for e in self:
                new_data_dict[e] = self[e] + other[e]
            return self.__class__(new_data_dict)
        else:
            raise Exception()

    def __sub__(self, other):
        r""""""
        if other.__class__ is self.__class__:
            new_data_dict = {}
            for e in self:
                new_data_dict[e] = self[e] - other[e]
            return self.__class__(new_data_dict)
        else:
            raise Exception()
