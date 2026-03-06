r"""


"""
import numpy as np
from scipy.sparse import csr_matrix

from phyem.tools.frozen import Frozen
from phyem.src.config import COMM, MASTER_RANK, RANK
from phyem.msehtt.tools.gathering_matrix import MseHttGatheringMatrix
from phyem.msehtt.tools.matrix.static.local import MseHttStaticLocalMatrix
from phyem.msehtt.tools.vector.static.local import MseHttStaticLocalVector


___gm0_cache___ = {
    'key': '',
    'gm0': None,
}


class Linear_System_Adjust(Frozen):
    r""""""

    def __init__(self, LS):
        r""""""
        self._LS = LS
        self._freeze()

    def locally_add_one_relation_and_one_Lagrangian_multiplier_to_the_end(
            self,
            arrays,
            b_value,
    ):
        r"""For example, if the current linear system is as follows:

        0     1     2         0
        3     4     5  =      1
        6     7     8         2

        arrays = [A, 0, B] where A, B vectors
        b_value = alpha  (a dictionary)

        Then the linear system will adjusted into, in element #e

        0     1     2     A[e]^T      0
        3     4     5     x       =   1
        6     7     8     B[e]^T      2
        A[e]  x     B[e]  x           alpha[e]

        where x is all zero.
        """
        # find colume gathering matrices ------------------------------------------
        gms = []
        for j in self._LS._CIA:
            gms.append(self._LS._CD[j])

        # define a row gathering matrix for only one row --------------------------
        key = gms[0]._signature

        if key != ___gm0_cache___['key']:
            local_element_indices = []
            for e in self._LS:
                local_element_indices.append(e)

            local_element_indices = COMM.gather(local_element_indices, root=MASTER_RANK)
            if RANK == MASTER_RANK:
                LOCAL = []
                for _ in local_element_indices:
                    LOCAL.extend(_)

                numbering = {}
                for i, e in enumerate(LOCAL):
                    numbering[e] = i
            else:
                numbering = None

            numbering = COMM.bcast(numbering, root=MASTER_RANK)

            row_gm = {}
            for e in self._LS:
                row_gm[e] = np.array([numbering[e], ])

            row_gm = MseHttGatheringMatrix(row_gm)
            # noinspection PyTypeChecker
            ___gm0_cache___['gm0'] = row_gm
            ___gm0_cache___['key'] = key
        else:
            row_gm = ___gm0_cache___['gm0']

        # -------- take care arrays ----------------------------------------------
        old_shape = self._LS._MIA.shape

        As = []   # to be put at positions in A of Ax=b
        for j in range(old_shape[1]):
            if isinstance(arrays[j], (int, float)) and arrays[j] == 0:
                A = MseHttStaticLocalMatrix(0, row_gm, gms[j])
            else:
                a = arrays[j]
                A = {}
                for e in self._LS:
                    ae = a[e]
                    if isinstance(ae, np.ndarray) and np.ndim(ae) == 1:
                        A[e] = csr_matrix(ae)
                    else:
                        raise NotImplementedError()

                A = MseHttStaticLocalMatrix(A, row_gm, gms[j])

            As.append(A)

        # ----------- take care of b_values ----------------------------------------
        if b_value == 0:
            b = MseHttStaticLocalVector(0, row_gm)  # to be put at positions in b of Ax=b
        else:
            v_dict = {}
            for e in self._LS:
                ve = b_value[e]
                v_dict[e] = np.array([ve, ])
            b = MseHttStaticLocalVector(v_dict, row_gm)
            # to be put at positions in b of Ax=b

        # -------------------------------------------------------------------------
        new_Shape = (old_shape[0]+1, old_shape[1]+1)

        new_MIA = - np.ones(new_Shape)
        new_VIA = - np.ones(new_Shape[0])
        new_RIA = - np.ones(new_Shape[0])
        new_CIA = - np.ones(new_Shape[1])
        new_MIA[:old_shape[0], :old_shape[1]] = self._LS._MIA
        new_VIA[:old_shape[0]] = self._LS._VIA
        new_RIA[:old_shape[0]] = self._LS._RIA
        new_CIA[:old_shape[1]] = self._LS._CIA

        j = max(self._LS._MD.keys())
        for k, A in enumerate(As):
            j += 1
            self._LS._MD[j] = A
            new_MIA[-1, k] = j

            j += 1
            self._LS._MD[j] = A.T
            new_MIA[k, -1] = j

        j += 1
        self._LS._MD[j] = MseHttStaticLocalMatrix(0, row_gm, row_gm)
        new_MIA[-1, -1] = j

        i = max(self._LS._VD.keys())
        self._LS._VD[i+1] = b
        new_VIA[-1] = i+1

        i = max(self._LS._RD.keys())
        self._LS._RD[i+1] = row_gm
        new_RIA[-1] = i+1

        i = max(self._LS._CD.keys())
        self._LS._CD[i+1] = row_gm
        new_CIA[-1] = i+1

        self._LS._MIA = new_MIA
        self._LS._VIA = new_VIA
        self._LS._RIA = new_RIA
        self._LS._CIA = new_CIA

        self._LS.check()
