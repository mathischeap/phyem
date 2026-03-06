r"""

"""
from scipy.sparse import isspmatrix_csc, isspmatrix_csr
import numpy as np
from scipy.sparse import bmat as sp_bmat

from phyem.tools.frozen import Frozen
from phyem.src.config import RANK, MASTER_RANK, COMM
from phyem.tools.linear_system.local.customizations.main import Linear_System_CUS
from phyem.tools.linear_system.local.adjust.main import Linear_System_Adjust
from phyem.tools.linear_system.local.assemble import General_Linear_System_Assemble


class General_Local_Linear_System(Frozen):
    r""""""

    def __init__(self, _2d_array_of_block_matrices, array_of_vectors, row_gms, col_gms=None):
        r"""
        Parameters
        ----------
        _2d_array_of_block_matrices :
            Be a 2d list or tuple or array that an entry is None or an object.
            If it is an object, the only request is that object[e] gives a sparse matrix for element #e.

        array_of_vectors :
            Be a list of vectors. The only request for the vector is that:
            vector[e] returns a 1d np.array for element #e.

        row_gms :
            The gathering matrices for the rows. The only request is that gm = row_gms[i]
            and gm[e] returns the numbering for element #e

        col_gms :
            The gathering matrices for the columes. The only request is that gm = row_gms[i]
            and gm[e] returns the numbering for element #e

        """
        I = len(_2d_array_of_block_matrices)     # I * J blocks
        J = len(_2d_array_of_block_matrices[0])  # I * J blocks
        assert len(array_of_vectors) == I, 'Block matrices and vectors shape dis-match.'
        for i in range(I):
            Mi = _2d_array_of_block_matrices[i]
            assert len(Mi) == J, f"The matrix block is wrong. It must be a rectangle."

        if col_gms is None:
            col_gms = row_gms
        else:
            pass

        assert len(row_gms) == I and len(col_gms) == J, 'Number of gathering matrices is wrong.'

        e = None

        for i in range(I):
            rgm = row_gms[i]
            Vi = array_of_vectors[i]
            for e in rgm:
                assert e in Vi, f"elements are wrong."
            Vi_e = Vi[e]
            assert isinstance(Vi_e, np.ndarray) and np.ndim(Vi_e) == 1, f"V[{i}][{e}] is not a 1d array."
            V_shape = Vi_e.shape[0]
            R_shape = len(rgm[e])

            for j in range(J):
                Mij = _2d_array_of_block_matrices[i][j]
                Mij_e = Mij[e]
                assert isspmatrix_csc(Mij_e) or isspmatrix_csr(Mij_e)

                M_shape = Mij_e.shape
                C_shape = len(col_gms[j][e])

                assert M_shape == (R_shape, C_shape) and V_shape == R_shape, f'Shape wrong for block[{i}][{j}]!'

        MIA = - np.ones([I, J])
        Mdict = dict()
        for i in range(I):
            for j in range(J):
                k = i * J + j
                MIA[i, j] = k
                Mdict[k] = _2d_array_of_block_matrices[i][j]

        VIA = - np.ones(I)
        Vdict = dict()
        for i in range(I):
            VIA[i] = i
            Vdict[i] = array_of_vectors[i]

        RIA = - np.ones(I)
        Rdict = dict()
        for i in range(I):
            RIA[i] = i
            Rdict[i] = row_gms[i]

        CIA = - np.ones(J)
        Cdict = dict()
        for j in range(J):
            CIA[j] = j
            Cdict[j] = col_gms[j]

        self._MIA = MIA
        self._MD = Mdict

        self._VIA = VIA
        self._VD = Vdict

        self._RIA = RIA
        self._RD = Rdict

        self._CIA = CIA
        self._CD = Cdict

        self._customizations_ = Linear_System_CUS(self)
        self._adjust_ = Linear_System_Adjust(self)

        self.check()

        self._freeze()

    # obtain the data for all elements after Customizations -----------------------------------------
    def M(self, i, j):
        r""""""
        return ___Mij___(self, i, j)

    def V(self, i):
        r""""""
        return ___Vi___(self, i)

    def RGM(self, i):
        r""""""
        return ___RGMi___(self, i)

    def CGM(self, j):
        r""""""
        return ___CGMj___(self, j)

    def check(self):
        r"""Do a self-check."""
        shape = self._MIA.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                M_indicator = self._MIA[i, j]
                assert M_indicator >= 0 and M_indicator in self._MD

            V_indicator = self._VIA[i]
            assert V_indicator >= 0 and V_indicator in self._VD

            R_indicator = self._RIA[i]
            assert R_indicator >= 0 and R_indicator in self._RD

        for j in range(shape[1]):
            C_indicator = self._CIA[j]
            assert C_indicator >= 0 and C_indicator in self._CD

    # --------------------------------------------------------------------------------------------------
    def __iter__(self):
        r"""Go through all local element indices"""
        for e in self._RD[list(self._RD.keys())[0]]:
            yield e

    def __contains__(self, e):
        r""""""
        return e in self._RD[list(self._RD.keys())[0]]

    # --------------------------------------------------------------------------------------------------
    @property
    def customize(self):
        r"""customize a particular block entry."""
        return self._customizations_

    @property
    def adjust(self):
        r"""adjust the linear system."""
        return self._adjust_

    @property
    def assemble(self):
        r"""We always make a new assemlber in case that the local system is changed."""
        return General_Linear_System_Assemble(self)

    def select_x(self, x, which):
        r"""Given a vector, we select from x and find those correspond to `which` unknowns.

        It returns a dict of local dofs and local cochains.
        """
        assert isinstance(which, list), f"put in list pls."
        gm = self.assemble._tCGM_
        local_vectors = {}
        for e in gm:
            local_dofs = gm[e]
            local_vectors[e] = x[local_dofs]

        gms = gm._gms

        local_x = {}

        for e in self:
            start = 0
            cochain = []
            for i, G in enumerate(gms):
                num_dofs = len(G[e])
                end = start + num_dofs
                if i in which:
                    local_dofs = local_vectors[e][start:end]
                    cochain.extend(local_dofs)
                else:
                    pass
                start = end
            assert start == len(gm[e])
            local_x[e] = np.array(cochain)

        return local_x

    # visualize ----------------------------------------------------------------------------------------
    def pr(self, e=None):
        r"""print the representation of the local linear system in element #e."""
        if e is None:
            if RANK == MASTER_RANK:
                e = None
                for e in self:
                    break
                assert e is not None, f"master rank must have an element."
            else:
                e = None
            e = COMM.bcast(e, root=MASTER_RANK)
        else:
            pass

        if e in self:
            pass
        else:
            return None

        import matplotlib.pyplot as plt
        import matplotlib
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "DejaVu Sans",
            "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
        })
        matplotlib.use('TkAgg')

        num_row, num_col = self._MIA.shape

        matrix_begin_text = r"\left[\begin{array}{" + r"c" * num_col + r"}"
        matrix_end_text = r"\end{array}\right]"

        A_text = matrix_begin_text
        for i in range(num_row):
            row_text = []
            for j in range(num_col):
                Mij = self.M(i, j)
                if Mij[e].nnz == 0:
                    Mij_text = r" "
                else:
                    Mij_text = rf" {Mij[e].shape}"
                row_text.append(Mij_text)
            row_text = ' & '.join(row_text)
            if i < num_row - 1:
                row_text += r" \\ "
            else:
                pass
            A_text += row_text
        A_text = A_text + matrix_end_text

        x_text = r"\begin{bmatrix}"
        for j in range(num_col):
            x_text += rf"x_{j}"
            if j < num_col - 1:
                x_text += r" \\ "
            else:
                pass

        x_text += r"\end{bmatrix}"

        b_text = r"\begin{bmatrix}"
        for i in range(num_row):
            b_text += str(self.V(i)[e].shape[0])

            if i < num_row - 1:
                b_text += r" \\ "
            else:
                pass

        b_text += r"\end{bmatrix}"

        text = r"$" + A_text + x_text + '=' + b_text + '$'
        fig=plt.figure(figsize=(10, 4))
        plt.axis((0, 1, 0, 1))
        plt.axis('off')
        plt.text(0.05, 0.47, text, ha='left', va='top', size=15)
        plt.title(rf'local linear system in element $\#${e}')
        plt.tight_layout()
        from phyem.src.config import _setting, _pr_cache
        if _setting['pr_cache']:
            _pr_cache(fig, filename='general_local_linear_system')
        else:
            plt.show(block=_setting['block'])

        return None

    def spy(self, e, markerfacecolor='k', markeredgecolor='g', markersize=6, threshold=None):
        r"""spy the local A (Ax=b) of rank element #e.

        Parameters
        ----------
        e
        markerfacecolor
        markeredgecolor
        markersize
        threshold

        Returns
        -------

        """
        if e in self:
            pass
        else:
            return None
        shape = self._MIA.shape
        A = [[] for _ in range(shape[0])]
        for i in range(shape[0]):
            for j in range(shape[1]):
                A[i].append(self.M(i, j)[e])

        M = sp_bmat(tuple(A), format='csr').toarray()

        if threshold is None:
            pass
        else:
            M[np.abs(M) < threshold] = 0

        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.spy(
            M,
            markerfacecolor=markerfacecolor,
            markeredgecolor=markeredgecolor,
            markersize=markersize
        )
        plt.tick_params(axis='both', which='major', direction='out')
        plt.tick_params(which='both', top=True, right=True, labelbottom=True, labelright=True)
        plt.show()
        return fig


# =======================================================================================================


class ___Mij___(Frozen):
    r""""""
    def __init__(self, LS, i, j):
        self._indicator = LS._MIA[i, j]
        self._meta = LS._MD[self._indicator]
        self._CUS = LS._customizations_
        self._freeze()

    def __getitem__(self, e):
        r""""""
        if self._indicator in self._CUS._M_:
            if e in self._CUS._M_[self._indicator]:
                return self._CUS._M_[self._indicator][e]
            else:
                return self._meta[e]
        else:
            return self._meta[e]


class ___Vi___(Frozen):
    r""""""
    def __init__(self, LS, i):
        self._indicator = LS._VIA[i]
        self._meta = LS._VD[self._indicator]
        self._CUS = LS._customizations_
        self._freeze()

    def __getitem__(self, e):
        r""""""
        if self._indicator in self._CUS._V_:
            if e in self._CUS._V_[self._indicator]:
                return self._CUS._V_[self._indicator][e]
            else:
                return self._meta[e]
        else:
            return self._meta[e]


class ___RGMi___(Frozen):
    r""""""
    def __init__(self, LS, i):
        self._indicator = LS._RIA[i]
        self._meta = LS._RD[self._indicator]
        self._CUS = LS._customizations_
        self._freeze()

    def __getitem__(self, e):
        r""""""
        if self._indicator in self._CUS._R_:
            if e in self._CUS._R_[self._indicator]:
                return self._CUS._R_[self._indicator][e]
            else:
                return self._meta[e]
        else:
            return self._meta[e]


class ___CGMj___(Frozen):
    r""""""
    def __init__(self, LS, j):
        self._indicator = LS._CIA[j]
        self._meta = LS._CD[self._indicator]
        self._CUS = LS._customizations_
        self._freeze()

    def __getitem__(self, e):
        r""""""
        if self._indicator in self._CUS._C_:
            if e in self._CUS._C_[self._indicator]:
                return self._CUS._C_[self._indicator][e]
            else:
                return self._meta[e]
        else:
            return self._meta[e]
