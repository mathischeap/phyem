r"""

"""
from scipy.sparse import csr_matrix

from phyem.tools.frozen import Frozen


class Linear_System_CUS(Frozen):
    r""""""

    def __init__(self, LS):
        self._LS = LS
        self._M_ = {}  # customizations that happen on the matrices
        # For example
        # self._M_ = {
        #   1: {
        #       5: sparse matrix,
        #       6: ...,
        #       ...
        #   }
        #   4: {
        #       ....
        #   }
        # }
        # This means the block matrix of indicator 1 is changed in the local elements #5, #6, ...
        self._V_ = {}  # customizations that happen on the vectors
        self._R_ = {}  # customizations that happen on the row gms
        self._C_ = {}  # customizations that happen on the col gms
        self._freeze()

    def replace_matrix_local_row(self, i, j, local_row_index, local_row_vector):
        r"""
        Replace the row #`local_row_index` of M[i][j] block matrix in element #e by
        `local_row_vector[e]`.

        """
        indicator = self._LS._MIA[i, j]

        M = self._LS.M(i, j)
        current_values = {}
        for e in self._LS:
            current_values[e] = M[e]

        target_dict = {}
        for e in self._LS:
            array = current_values[e].tolil()
            array[local_row_index, :] = local_row_vector[e]
            target_dict[e] = csr_matrix(array)
        self._M_[indicator] = target_dict

    def replace_vector_local_value(self, i, local_row_index, local_value):
        r"""
        Replace the `local_row_index`th vale of of V[i] block vector in element #e  by
        `local_row_vector[e]`.

        """
        indicator = self._LS._VIA[i]

        V = self._LS.V(i)
        current_values = {}
        for e in self._LS:
            current_values[e] = V[e]

        target_dict = {}
        for e in self._LS:
            array = current_values[e]
            array[local_row_index] = local_value[e]
            target_dict[e] = array

        self._V_[indicator] = target_dict
