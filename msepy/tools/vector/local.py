

from tools.frozen import Frozen


class LocalVector(Frozen):
    """"""
    def __init__(self, _2d_data, gathering_matrix):
        """"""
        self._data = _2d_data  # 2d numpy array or csr-sparse-matrix or None.
        self._gm = gathering_matrix
        self._freeze()

    def __getitem__(self, i):
        """When we receive None for data, this raise Error."""
        return self._data[i]
