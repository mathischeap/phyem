# -*- coding: utf-8 -*-
r"""
"""
import matplotlib.pyplot as plt
import matplotlib

from src.config import RANK, MASTER_RANK, SIZE, COMM

from tools.frozen import Frozen
from _MPI.generic.py.matrix.localize.static import (MPI_PY_Localize_Static_Matrix,
                                                    MPI_PY_Localize_Static_Vector,
                                                    MPI_PY_Localize_Static_Vector_Cochain)

from _MPI.generic.py.matrix.localize.bmat import bmat
from _MPI.generic.py.vector.localize.concatenate import concatenate

from _MPI.generic.py.linear_system.localize.static.customize import _MPI_PY_LS_Customize

from _MPI.generic.py.linear_system.globalize.static import MPI_PY_Globalize_Static_Linear_System


class MPI_PY_Localize_Static_Linear_System(Frozen):
    """"""

    def __init__(self, A, x, b, _pr_texts=None, _time_indicating_text=None, _str_args=''):
        """Note that I am not receiving any BC. all BC must already be included in A, x, b."""
        row_shape = len(A)
        col_shape = len(A[0])
        assert len(x) == col_shape and len(b) == row_shape, "A, x, b shape dis-match."
        self._shape = (row_shape, col_shape)
        self._parse_gathering_matrices(A, x, b)
        self._A = _AAA(self, A)   # A is a 2d list of MsePyStaticLocalMatrix
        self._x = _Xxx(self, x)   # x ia a list of MsePyStaticLocalVector (or subclass)
        self._b = _Bbb(self, b)   # b ia a list of MsePyStaticLocalVector (or subclass)
        self._customize = None

        if RANK == MASTER_RANK:
            self._pr_texts = _pr_texts
            self._time_indicating_text = _time_indicating_text
            self._str_args = _str_args
        else:
            pass

        self._freeze()

    @property
    def shape(self):
        return self._shape

    def _parse_gathering_matrices(self, A, x, b):
        """"""
        row_shape, col_shape = self.shape
        row_gms = [None for _ in range(row_shape)]
        col_gms = [None for _ in range(col_shape)]

        for i in range(row_shape):
            for j in range(col_shape):
                A_ij = A[i][j]

                if A_ij is None:
                    pass
                else:
                    assert A_ij.__class__ is MPI_PY_Localize_Static_Matrix, f"A[{i}][{j}] is {A_ij.__class__}, wrong!"
                    row_gm_i = A_ij._gm_row
                    col_gm_j = A_ij._gm_col

                    if row_gms[i] is None:
                        row_gms[i] = row_gm_i
                    else:
                        assert row_gms[i] is row_gm_i, \
                            f"by construction, this must be the case as we only construct" \
                            f"gathering matrix once and store only once copy somewhere!"

                    if col_gms[j] is None:
                        col_gms[j] = col_gm_j
                    else:
                        assert col_gms[j] is col_gm_j, \
                            f"by construction, this must be the case as we only construct" \
                            f"gathering matrix once and store only once copy somewhere!"

        assert None not in row_gms and None not in col_gms, f"miss some gathering matrices."

        self._row_gms = row_gms
        self._col_gms = col_gms

        # now we check gathering matrices in x.
        for j in range(col_shape):
            x_j = x[j]
            assert x_j.__class__ is MPI_PY_Localize_Static_Vector_Cochain, f"x[{j}] is {x_j.__class__}, wrong"
            gm_j = x_j._gm
            assert gm_j == self._col_gms[j]

        # now we check gathering matrices in b.
        for i in range(row_shape):
            b_i = b[i]
            if b_i is None:
                pass
            else:
                assert b_i.__class__ in (MPI_PY_Localize_Static_Vector, MPI_PY_Localize_Static_Vector_Cochain), \
                    f"b[{i}] is {b_i.__class__}, wrong"
                gm_i = b_i._gm
                assert gm_i == self._row_gms[i]

    @property
    def gathering_matrices(self):
        """return all gathering matrices; both row and col.

        Note the differences to the `gm0_row` and `gm1_col` of `self.A._mA`.
        """
        return self._row_gms, self._col_gms

    @property
    def global_gathering_matrices(self):
        """Notice the difference from ``self.gathering_matrices``."""
        return self.A._mA._gm_row, self.A._mA._gm_col

    @property
    def A(self):
        """``Ax = b``

        A is a 2-d list of None or msepy static local matrices.
        """
        return self._A

    @property
    def x(self):
        """``Ax = b``

        x is a 1d list of msepy static root-form cochain vector.
        """
        return self._x

    @property
    def b(self):
        """``Ax = b``

        b is a 1d list of msepy static local vectors.
        """
        return self._b

    def assemble(self, cache=None):
        """

        Parameters
        ----------
        cache : {None, str}, default, None
            We can manually cache the assembled A matrix by set ``cache`` to be a string. When next time
            it sees the same `cache` it will return the cached A matrix from the cache.

            This is very helpful, for example, when the A matrix does not change in all iterations.

        Returns
        -------

        """
        A = self.A._mA.assemble(cache=cache)
        b = self.b._vb.assemble()
        ALS = MPI_PY_Globalize_Static_Linear_System(A, b)
        return ALS

    @property
    def customize(self):
        """customize A and b."""
        if self._customize is None:
            self._customize = _MPI_PY_LS_Customize(self)
        return self._customize

    def pr(self):
        """"""
        if RANK != MASTER_RANK:
            return
        else:
            pass

        if self._pr_texts is None:
            print('No texts to print.')
            return

        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "DejaVu Sans",
            "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
        })
        matplotlib.use('TkAgg')

        texts = self._pr_texts
        A_text, x_text, b_text = texts
        I_ = len(A_text)
        _J = len(A_text[0])
        tA = ''
        tx = ''
        tb = ''
        for i in range(I_):
            for j in range(_J):
                tA_ij = A_text[i][j]
                if tA_ij == '':
                    tA_ij = '0'
                else:
                    pass
                tA += tA_ij
                if j < _J - 1:
                    tA += '&'
            if i < I_ - 1:
                tA += r'\\'
        tA = r"\begin{bmatrix}" + tA + r"\end{bmatrix}"
        for j in range(_J):
            tx_j = x_text[j]
            assert tx_j != '', f"unknown must be something!"
            tx += tx_j
            if j < _J - 1:
                tx += r'\\'
        tx = r"\begin{bmatrix}" + tx + r"\end{bmatrix}"
        for i in range(I_):
            tb_i = b_text[i]
            if tb_i == '':
                tb_i = '0'
            tb += tb_i
            if i < I_ - 1:
                tb += r'\\'
        tb = r"\begin{bmatrix}" + tb + r"\end{bmatrix}"

        text = tA + tx + '=' + tb
        text = r"$" + text + r"$"
        fig = plt.figure(figsize=(10, 4))
        plt.axis((0, 1, 0, 1))
        plt.axis('off')
        if self._time_indicating_text is None:
            plt.text(0.05, 0.5, text, ha='left', va='center', size=15)
        else:
            plt.text(0.05, 0.525, text, ha='left', va='bottom', size=15)
            plt.plot(
                [0, 1], [0.5, 0.5], '--', color='gray', linewidth=0.5,
            )
            A_text, x_text, b_text = self._time_indicating_text
            tA = ''
            tx = ''
            tb = ''
            for i in range(I_):
                for j in range(_J):
                    tA_ij = A_text[i][j]
                    if tA_ij == '':
                        tA_ij = r'\divideontimes'
                    else:
                        pass
                    tA += tA_ij
                    if j < _J - 1:
                        tA += '&'
                if i < I_ - 1:
                    tA += r'\\'
            tA = r"\begin{bmatrix}" + tA + r"\end{bmatrix}"
            for j in range(_J):
                tx_j = x_text[j]
                if tx_j == '':
                    tx_j = r'\divideontimes'
                tx += tx_j
                if j < _J - 1:
                    tx += r'\\'
            tx = r"\begin{bmatrix}" + tx + r"\end{bmatrix}"
            for i in range(I_):
                tb_i = b_text[i]
                if tb_i == '':
                    tb_i = r'\divideontimes'
                tb += tb_i
                if i < I_ - 1:
                    tb += r'\\'
            tb = r"\begin{bmatrix}" + tb + r"\end{bmatrix}"
            text = tA + tx + '=' + tb
            text = r"$" + text + r"$"
            plt.text(0.05, 0.47, text, ha='left', va='top', size=15)
        plt.title(f'MPI-PY-Local-Linear-System evaluated @ [{self._str_args}]')
        plt.tight_layout()
        from src.config import _setting, _pr_cache
        if _setting['pr_cache']:
            _pr_cache(fig, filename='mpi_py_msehy_staticLocalLinearSystem')
        else:
            plt.show(block=_setting['block'])

        return fig


class _AAA(Frozen):
    """"""

    def __init__(self, ls, A):
        self._ls = ls
        self._mA = bmat(A)
        self._A = A  # save the block-wise A
        self._freeze()

    def __iter__(self):
        """go through all local elements."""
        for index in range(self._mA):
            yield index

    def __getitem__(self, i):
        """Get the local system for element #i. Adjustments and customizations will take effect."""
        return self._mA[i]

    def spy(self, i, **kwargs):
        """spy plot the local A matrix of element #i."""
        return self._mA.spy(i, **kwargs)


class _Xxx(Frozen):
    """"""
    def __init__(self, ls, x):
        time_slots = list()
        for c in range(ls.shape[1]):
            assert x[c].__class__ is MPI_PY_Localize_Static_Vector_Cochain, \
                f"x[{c}] is not a static cochain vector."
            time_slots.append(
                x[c]._t
            )
        self._representing_time = time_slots
        self._ls = ls
        self._x = x  # important!, use to override cochain back to the form.
        self._vx = concatenate(x, ls.A._mA._gm_col)

    @property
    def representing_time(self):
        """The cochain static vector are at these time instants for the corresponding root-forms."""
        return self._representing_time

    def update(self, x):
        """# update the unknowns from x (usually the solution)."""

        if SIZE == 1:
            self._parallel_update(x)

        else:
            None_x = x is None
            None_x_ranks = COMM.allgather(None_x)
            if None_x_ranks.count(True) == 0:
                # we got no None in all ranks, then just parallelize it.
                self._parallel_update(x)
            elif None_x_ranks.count(False) == 1:
                # we only find the solution in one single core.
                solution_rank = None_x_ranks.index(False)
                x = COMM.bcast(x, root=solution_rank)
                self._parallel_update(x)
            else:
                raise NotImplementedError()

    def _parallel_update(self, x):
        """"""
        gm = self._vx._gm
        data_dict = dict()
        for index in gm:
            gmi = gm[index]
            data_dict[index] = x[gmi]
        gms = gm._gms
        num_gm = len(gms)
        individual_cochain = list()
        for _ in range(num_gm):
            individual_cochain.append(
                dict()
            )
        for index in gm:
            num_dof = 0
            total_cochain = data_dict[index]
            for j, _gms in enumerate(gms):
                num_local_dofs = _gms.num_local_dofs(index)
                cochain = total_cochain[num_dof:num_dof+num_local_dofs]
                num_dof += num_local_dofs
                individual_cochain[j][index] = cochain

        for i, cochain in enumerate(individual_cochain):
            self._x[i]._set_data(cochain)
            self._x[i].override()


class _Bbb(Frozen):
    """"""
    def __init__(self, ls, b):
        self._ls = ls
        self._b = b  # save the block-wise b
        self._vb = concatenate(b, ls.A._mA._gm_row)

    def __getitem__(self, i):
        """Get the local vector for element `#i`."""
        return self._vb[i]
