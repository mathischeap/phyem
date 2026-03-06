r""""""

from phyem.tools.frozen import Frozen
from phyem.msehtt.tools.gathering_matrix import MseHttGatheringMatrix

from phyem.msehtt.tools.matrix.static.local import MseHttStaticLocalMatrix
from phyem.msehtt.tools.vector.static.local import MseHttStaticLocalVector
from phyem.msehtt.tools.matrix.static.local import bmat as msehtt_bmat
from phyem.msehtt.tools.vector.static.local import concatenate as msehtt_concatenate

from phyem.tools.linear_system.global_.main import General_Global_Linear_System


class General_Linear_System_Assemble(Frozen):
    r""""""

    def __init__(self, LS):
        r""""""
        self._LS = LS
        self._tRGM_ = self._make_total_gathering_matrix(LS._RIA, LS._RD)  # total row gathering matrix
        self._tCGM_ = self._make_total_gathering_matrix(LS._CIA, LS._CD)  # total col gathering matrix
        self._freeze()

    @classmethod
    def _make_total_gathering_matrix(cls, indicator_array, gm_dict):
        r""""""
        gms = []
        names = []
        for indicator in indicator_array:
            gm = gm_dict[indicator]
            gms.append(gm)
            names.append(gm.__class__.__name__)

        if all([_ == MseHttGatheringMatrix.__name__ for _ in names]):
            tGM = MseHttGatheringMatrix(gms)
        else:
            raise NotImplementedError()

        return tGM

    def __call__(self, FORMAT='csc', threshold=None):
        r""""""
        # -- pick up gathering matrices -----------------------------------------
        gms_row = []
        for i in self._LS._RIA:
            gms_row.append(self._LS._RD[i])

        gms_col = []
        for j in self._LS._CIA:
            gms_col.append(self._LS._CD[j])

        # -----------------------------------------------------------------------
        shape = self._LS._MIA.shape

        # -------- bmat A -------------------------------------------------------
        A_list = [[] for _ in range(shape[0])]
        mtypes = []
        for i in range(shape[0]):
            for j in range(shape[1]):

                M = self._LS.M(i, j)
                value_dict = {}
                for e in self._LS:
                    value_dict[e] = M[e]

                M = self._sheid_M_(value_dict, gms_row[i], gms_col[j])

                # indicator = self._LS._MIA[i, j]
                # M = self._LS._MD[indicator]
                A_list[i].append(M)
                mtypes.append(M.__class__.__name__)

        if all([_ == MseHttStaticLocalMatrix.__name__ for _ in mtypes]):
            A = msehtt_bmat(A_list)
        else:
            raise NotImplementedError()

        # ----------- concatenate b ----------------------------------------------------
        b_list = []
        btypes = []
        for i in range(shape[0]):

            V = self._LS.V(i)
            value_dict = {}
            for e in self._LS:
                value_dict[e] = V[e]

            V = self._sheid_V_(value_dict, gms_row[i])

            # V = self._LS._VD[indicator]
            b_list.append(V)
            btypes.append(V.__class__.__name__)

        if all([_ == MseHttStaticLocalVector.__name__ for _ in btypes]):
            b = msehtt_concatenate(b_list, self._tRGM_)
        else:
            raise NotImplementedError()

        # --------------- assemble and solve Ax=b ---------------------------------------
        global_A = A.assemble(FORMAT=FORMAT, threshold=threshold)
        global_b = b.assemble(vtype='distributed', mode='sum')

        return General_Global_Linear_System(global_A, global_b)

    @staticmethod
    def _sheid_M_( M_value_dict, gm_row, gm_col):
        r""""""
        return MseHttStaticLocalMatrix(
            M_value_dict, gm_row, gm_col
        )

    @staticmethod
    def _sheid_V_(V_value_dict, gm_row):
        r""""""
        return MseHttStaticLocalVector(
            V_value_dict, gm_row
        )
