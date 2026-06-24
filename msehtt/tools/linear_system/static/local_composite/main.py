# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
# from scipy.sparse import isspmatrix_csr, isspmatrix_csc
from phyem.src.config import RANK, MASTER_RANK
from scipy.sparse import csc_matrix, csr_matrix
from phyem.tools.frozen import Frozen
from phyem.msehtt.tools.linear_system.static.global_.main import MseHttLinearSystem
from phyem.msehtt.tools.linear_system.static.local.main import MseHttStaticLocalLinearSystem
from phyem.msehtt.tools.matrix.static.global_ import MseHttGlobalMatrix
from phyem.msehtt.tools.vector.static.global_distributed import MseHttGlobalVectorDistributed


class MseHtt_Static_Local_Composite_LinearSystem(Frozen):
    """It takes several msehtt local linear systems as inputs. And also we need to mention in each element,
    which local linear system is eventually used.

    This is very useful when, for example, in different elements, we have different discretizations. In this case,
    we can firstly set up local linear system in all elements and then select one for each element.

    """

    def __init__(self, *static_local_linear_systems, condlist=None):
        r"""
        Parameters
        ----------
        static_local_linear_systems :
            The local systems to be used.
        conditions :
            The conditions that determine in each element which local system should be selected.
        """
        GMs_row = list()
        GMs_col = list()
        for i, obj in enumerate(static_local_linear_systems):
            assert obj.__class__ in (MseHttStaticLocalLinearSystem, ), f"{i}th object is not a valid object."
            gathering_matrices = obj.global_gathering_matrices
            GMs_row.append(gathering_matrices[0])
            GMs_col.append(gathering_matrices[1])

        for i, gm in enumerate(GMs_row):
            assert gm == GMs_row[0], f"row gathering matrix {i}th linear system does not match."
        for i, gm in enumerate(GMs_col):
            assert gm == GMs_col[0], f"col gathering matrix {i}th linear system does not match."

        self._global_row_gathering_matrix = GMs_row[0]
        self._global_col_gathering_matrix = GMs_col[0]

        self._static_local_linear_systems = static_local_linear_systems
        self._condlist = condlist
        self._system_usage_dict_ = None

        # ------ check x of all static local systems -------
        X = None
        for k, sls in enumerate(static_local_linear_systems):
            variables = sls.x._x
            if k == 0:
                X = variables
            else:
                for x, v in zip(X, variables):
                    assert x._f is v._f, \
                        (f"variables in {k}th static local linear system does not match "
                         f"that of 0th static local linear system.")

        self._freeze()

    @property
    def condlist(self):
        r""""""
        assert self._condlist is not None, f"condition list is not set yet."
        return self._condlist

    @property
    def system_usage_dict(self):
        if self._system_usage_dict_ is None:
            system_usage_dict = dict()
            # system_usage_dict[i] = k means in local element #i,
            # we use local system of {k}th static local linear system
            conditions = self.condlist
            for i in self._global_row_gathering_matrix:  # go through all local element indices
                for k, cond in enumerate(conditions):
                    ToF = cond(i)
                    if ToF:
                        system_usage_dict[i] = k
                        break
                    else:
                        pass
                assert i in system_usage_dict, f"Cannot find a local system for element #{i}. Check the condition list."
            self._system_usage_dict_ = system_usage_dict

        return self._system_usage_dict_

    @property
    def x(self):
        r"""We have checked the consistence in the __init__ function.
        So just return x of the frist static local linear system.
        """
        return self._static_local_linear_systems[0].x

    def assemble(self, FORMAT='csc', preconditioner=None, threshold=None, customizations=None):
        r"""

        """
        if preconditioner is None:
            pass
        else:
            pass

        # --------------------------------------------------------------------------------------------------------------

        A_customizations = []
        b_customizations = []

        if customizations is None:
            pass
        else:
            for cus in customizations:
                for key in cus:
                    assert key in ('A', 'b'), \
                        f"each set of customization can only customize A and b, or, A or b."

                if 'A' in cus:
                    cus_A = cus['A']
                    indicator = cus_A[0]
                    if indicator == 'new_EndZeroRowCol_with_a_one_for_global_dof':
                        ith_unknown, global_dof = cus_A[1], cus_A[2]
                        A_customizations.append(
                            ('new_EndZeroRowCol_with_a_one_for_global_dof', ith_unknown, global_dof)
                        )
                    else:
                        raise NotImplementedError(indicator)
                else:
                    pass

                if 'b' in cus:
                    cus_b = cus['b']
                    indicator = cus_b[0]
                    if indicator == 'add_a_value_at_the_end':
                        value = cus_b[1]
                        b_customizations.append(('add_a_value_at_the_end', value))
                    else:
                        raise NotImplementedError(indicator)
                else:
                    pass

        if len(A_customizations) == 0:
            A_customizations = None
        else:
            pass

        if len(b_customizations) == 0:
            b_customizations = None
        else:
            pass

        # -----------------------------------------------------------------------------------------------------------

        system_usage_dict = self.system_usage_dict
        gm_row = self._global_row_gathering_matrix
        gm_col = self._global_col_gathering_matrix

        # --- Now, we do the assembling: A -----------------------------------------------------------------------------

        if A_customizations is not None:
            A = self.___A_customized_call___(
                FORMAT, threshold, A_customizations,
            )
        else:
            ROW = list()
            COL = list()
            DAT = list()

            for i in self._global_row_gathering_matrix:  # go through all local element indices
                k = system_usage_dict[i]
                static_local_system = self._static_local_linear_systems[k]
                Mi = static_local_system.A._mA[i]  # all adjustments and customizations take effect
                indices = Mi.indices
                indptr = Mi.indptr
                data = Mi.data
                if threshold is None:
                    pass
                else:
                    data[np.abs(data) < threshold] = 0.

                nums: list = list(np.diff(indptr))
                row = []
                col = []

                if Mi.__class__.__name__ == 'csc_matrix':
                    for j, num in enumerate(nums):
                        idx = indices[indptr[j]:indptr[j+1]]
                        row.extend(gm_row[i][idx])
                        col.extend([gm_col[i][j] for _ in range(num)])

                elif Mi.__class__.__name__ == 'csr_matrix':
                    for j, num in enumerate(nums):
                        idx = indices[indptr[j]:indptr[j+1]]
                        row.extend([gm_row[i][j] for _ in range(num)])
                        col.extend(gm_col[i][idx])

                else:
                    raise Exception("I can not handle %r." % Mi)

                ROW.extend(row)
                COL.extend(col)
                DAT.extend(data)

            if FORMAT == 'csc':
                SPA_MATRIX = csc_matrix
            elif FORMAT == 'csr':
                SPA_MATRIX = csr_matrix
            else:
                raise Exception

            dep = int(gm_row.num_global_dofs)
            wid = int(gm_col.num_global_dofs)

            A = SPA_MATRIX((DAT, (ROW, COL)), shape=(dep, wid))
            A = MseHttGlobalMatrix(A, gm_row, gm_col)

            del DAT, ROW, COL

        # --- Next, we assemble b --------------------------------------------------------------------------------------
        v = np.zeros(gm_row.num_global_dofs)
        for i in gm_row:
            k = system_usage_dict[i]
            static_local_system = self._static_local_linear_systems[k]
            v[gm_row[i]] += static_local_system.b._vb[i]  # must do this to be consistent with the matrix assembling.

        b = MseHttGlobalVectorDistributed(v, gm_row)

        if b_customizations is None:
            pass
        else:
            b = self._deal_with_b_customizations_(b, b_customizations, vtype='distributed', mode='sum')

        # ==============================================================================================================
        return MseHttLinearSystem(A, b)

    # ------------ CUSTOMIZED A ----------------------------------------------------------------------------------------

    def ___A_customized_call___(self, FORMAT, threshold, customizations):
        r""" Here, we will do some customizations to the assembled A matrix.

        Parameters
        ----------
        FORMAT
        threshold
        customizations :


        Returns
        -------

        """
        if len(customizations) == 1:
            cus = customizations[0]
            indicator = cus[0]
            if indicator == "new_EndZeroRowCol_with_a_one_for_global_dof":
                ith_unknown, global_dof = cus[1], cus[2]
                # the place of the new `1` entry is the `global_dof` of `ith_unknown`.
                return self.___new_EndZeroRowCol_with_a_one_for_global_dof___(
                    FORMAT, threshold, ith_unknown, global_dof
                )

            elif indicator == "new_EndZeroRowCol_with_a_one_for_local_dof":
                ith_unknown, element_index, local_dof = cus[1], cus[2], cus[3]
                return self.___new_EndZeroRowCol_with_a_one_for_local_dof___(
                    FORMAT, threshold, ith_unknown, element_index, local_dof
                )
            else:
                raise NotImplementedError(
                    f"indicator={indicator} of ___customized_call___ of {self.__class__} is not coded!"
                )
        else:
            raise NotImplementedError(
                f"___customized_call___ of {self.__class__} for more customizations is not coded."
            )

    def ___new_EndZeroRowCol_with_a_one_for_global_dof___(
            self, FORMAT, threshold,
            ith_unknown, global_dof
    ):
        r"""When the assembling only have one customization and this customization is to
        add a new line at the end who only have zero-entries except that there is one `1` at the
        place for the `global_dof` of `ith_unknown`.

        Parameters
        ----------
        FORMAT
        threshold
        ith_unknown
        global_dof

        Returns
        -------

        """
        gm_col = self._global_row_gathering_matrix
        place = gm_col.find_global_numbering_of_ith_composition_global_dof(ith_unknown, global_dof)
        return self._new_EndZeroRowCol_with_a_one_(FORMAT, threshold, place)

    def ___new_EndZeroRowCol_with_a_one_for_local_dof___(
            self, FORMAT, threshold,
            ith_unknown, element_index, local_dof
    ):
        r""""""
        gm_col = self._global_row_gathering_matrix
        place = gm_col.find_global_numbering_of_ith_composition_local_dof(ith_unknown, element_index, local_dof)
        return self._new_EndZeroRowCol_with_a_one_(FORMAT, threshold, place)

    def _new_EndZeroRowCol_with_a_one_(self, FORMAT, threshold, place):
        r""""""

        gm_row = self._global_row_gathering_matrix
        gm_col = self._global_col_gathering_matrix
        system_usage_dict = self.system_usage_dict

        ROW = list()
        COL = list()
        DAT = list()

        # A = SPA_MATRIX((dep, wid))  # initialize a sparse matrix

        for i in self._global_row_gathering_matrix:  # go through all local element indices
            k = system_usage_dict[i]
            static_local_system = self._static_local_linear_systems[k]
            Mi = static_local_system.A._mA[i]  # all adjustments and customizations take effect
            indices = Mi.indices
            indptr = Mi.indptr
            data = Mi.data
            if threshold is None:
                pass
            else:
                data[np.abs(data) < threshold] = 0.

            nums: list = list(np.diff(indptr))
            row = []
            col = []

            if Mi.__class__.__name__ == 'csc_matrix':
                for j, num in enumerate(nums):
                    idx = indices[indptr[j]:indptr[j+1]]
                    row.extend(gm_row[i][idx])
                    col.extend([gm_col[i][j] for _ in range(num)])

            elif Mi.__class__.__name__ == 'csr_matrix':
                for j, num in enumerate(nums):
                    idx = indices[indptr[j]:indptr[j+1]]
                    row.extend([gm_row[i][j] for _ in range(num)])
                    col.extend(gm_col[i][idx])

            else:
                raise Exception("I can not handle %r." % Mi)

            ROW.extend(row)
            COL.extend(col)
            DAT.extend(data)

        if FORMAT == 'csc':
            SPA_MATRIX = csc_matrix
        elif FORMAT == 'csr':
            SPA_MATRIX = csr_matrix
        else:
            raise Exception

        dep = int(gm_row.num_global_dofs)
        wid = int(gm_col.num_global_dofs)

        if RANK == MASTER_RANK:
            DAT.append(1)
            ROW.append(dep)
            COL.append(place)
            DAT.append(1)
            ROW.append(place)
            COL.append(wid)
        else:
            pass
        A = SPA_MATRIX((DAT, (ROW, COL)), shape=(dep+1, wid+1))
        A = MseHttGlobalMatrix(A)

        assert A.value_at(dep, place) == 1, r"must be!"
        assert A.value_at(place, wid) == 1, r"must be!"
        assert A.nnz_of_row(dep) == 1, r"must be!"
        assert A.nnz_of_col(wid) == 1, r"must be!"

        return A

    # --- CUSTOMIZATIONS of b ------------------------------------------------------------------------------------------

    @staticmethod
    def _deal_with_b_customizations_(RETURN, customizations, vtype, mode):
        r""""""
        if len(customizations) == 1:
            cus = customizations[0]
            indicator = cus[0]
            if indicator == 'add_a_value_at_the_end':
                value = cus[1]
                if vtype == 'distributed' and mode == 'sum' and value == 0:
                    new_v = np.append(RETURN.V, [value, ])
                    return MseHttGlobalVectorDistributed(new_v)
                else:
                    raise NotImplementedError(f"vtype={vtype}, mode={mode}, value={value} not implemented.")
            else:
                raise NotImplementedError(
                    f"MseHttStaticLocalVectorAssemble _deal_with_customizations_ cannot do for "
                    f"indicator={indicator}."
                )
        else:
            raise NotImplementedError(
                f"MseHttStaticLocalVectorAssemble cannot deal with multi customizations yet"
            )
