# -*- coding: utf-8 -*-
r"""
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from phyem.src.config import RANK, MASTER_RANK, COMM, MPI

from phyem.tools.frozen import Frozen
from phyem.msehtt.tools.matrix.static.local import MseHttStaticLocalMatrix
from phyem.msehtt.tools.vector.static.local import MseHttStaticLocalVector
from phyem.msehtt.static.form.cochain.vector.static import MseHttStaticCochainVector

from phyem.msehtt.tools.matrix.static.local import bmat
from phyem.msehtt.tools.vector.static.local import concatenate

from phyem.msehtt.tools.linear_system.static.global_.main import MseHttLinearSystem
from phyem.msehtt.tools.linear_system.static.local.customize import MseHttStaticLinearSystemCustomize

from phyem.msehtt.tools.linear_system.static.local.solve import MseHtt_Local_LinearSystem_Solve


class MseHttStaticLocalLinearSystem(Frozen):
    """"""

    def __init__(self, A, x, b, _pr_texts=None, _time_indicating_text=None, _str_args=''):
        """"""
        row_shape = len(A)
        col_shape = len(A[0])
        assert len(x) == col_shape and len(b) == row_shape, "A, x, b shape dis-match."
        self._shape = (row_shape, col_shape)
        self._parse_gathering_matrices(A, x, b)
        self._A = _AAA(self, A)   # A is a 2d list of MsePyStaticLocalMatrix
        self._x = _Xxx(self, x)   # x ia a list of MsePyStaticLocalVector (or subclass)
        self._b = _Bbb(self, b)   # b ia a list of MsePyStaticLocalVector (or subclass)
        self._customize = None
        if RANK == MASTER_RANK:  # for pr purpose, only saved in the master rank.
            self._pr_texts = _pr_texts
            self._time_indicating_text = _time_indicating_text
            self._str_args = _str_args
        else:
            pass
        self._solve = None
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
                    assert A_ij.__class__ is MseHttStaticLocalMatrix, f"A[{i}][{j}] is {A_ij.__class__}, wrong!"
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
        num_elements = list()
        for rgm in row_gms:
            # noinspection PyUnresolvedReferences
            num_elements.append(rgm.num_rank_elements)
        for cgm in col_gms:
            # noinspection PyUnresolvedReferences
            num_elements.append(cgm.num_rank_elements)
        # noinspection PyTypeChecker
        assert all(np.array(num_elements) == num_elements[0]), f"total element number dis-match"
        self._num_rank_elements = num_elements[0]
        self._row_gms = row_gms
        self._col_gms = col_gms
        for gm in row_gms:
            # noinspection PyUnresolvedReferences
            assert gm.num_rank_elements == self._num_rank_elements, f'must be'
        for gm in col_gms:
            # noinspection PyUnresolvedReferences
            assert gm.num_rank_elements == self._num_rank_elements, f'must be'

        # now we check gathering matrices in x.
        for j in range(col_shape):
            x_j = x[j]
            assert x_j.__class__ is MseHttStaticCochainVector, f"x[{j}] is {x_j.__class__}, wrong"
            gm_j = x_j._gm
            assert gm_j is self._col_gms[j], \
                f"by construction, this must be the case as we only construct" \
                f"gathering matrix once and store only once copy somewhere!"

        # now we check gathering matrices in b.
        for i in range(row_shape):
            b_i = b[i]
            if b_i is None:
                pass
            else:
                assert b_i.__class__ in (MseHttStaticLocalVector, MseHttStaticCochainVector), \
                    f"b[{i}] is {b_i.__class__}, wrong"
                gm_i = b_i._gm
                assert gm_i is self._row_gms[i], \
                    f"by construction, this must be the case as we only construct" \
                    f"gathering matrix once and store only once copy somewhere!"

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
    def num_rank_elements(self):
        """amount of elements in this rank"""
        return self._num_rank_elements

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

    def __iter__(self):
        """Go through all local elements."""
        for i in self.A._mA:
            yield i

    def spy(self, e, **kwargs):
        """spy plot the A matrix of local element #e."""
        return self.A.spy(e, **kwargs)

    def condition_number(self, e):
        """Return the condition of number of the local system in element #e."""
        return self.A.condition_number(e)

    def rank(self, e):
        """Return the condition of number of the local system in element #e."""
        return self.A.rank(e)

    def num_singularities(self, e):
        """Return the condition of number of the local system in element #e."""
        return self.A.num_singularities(e)

    @property
    def solve(self):
        """"""
        if self._solve is None:
            self._solve = MseHtt_Local_LinearSystem_Solve(self)
        return self._solve

    def assemble(self, cache=None, preconditioner=False, threshold=None, customizations=None):
        """

        Parameters
        ----------
        cache : {None, str}, default, None
            We can manually cache the assembled A matrix by set ``cache`` to be a string. When next time
            it sees the same `cache` it will return the cached A matrix from the cache.

            This is very helpful, for example, when the A matrix does not change in all iterations.
        preconditioner :
        threshold :
        customizations :
            The customizations that need to be done in the assembled system.

            customizations = [
                {
                    'A': ...,
                    'b': ...
                },
                {
                    'A': ...,
                    'b': ...
                }
            ]

            So, each customization is a dict, and it can have two keys that contain the customization
            sent to A or b.

        Returns
        -------

        """
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

        A = self.A._mA.assemble(cache=cache, threshold=threshold, customizations=A_customizations)
        b = self.b._vb.assemble(customizations=b_customizations)

        # noinspection PySimplifyBooleanCheck
        if preconditioner is False:  # this is not a typo; do NOT use: if not preconditioner:
            pass
        elif preconditioner == 'Jacobian':
            diag = A.diagonal()
            A = diag @ A
            b = diag @ b
        else:
            raise NotImplementedError(f"preconditioner={preconditioner}")
        return MseHttLinearSystem(A, b)

    @property
    def customize(self):
        """customize A and b."""
        if self._customize is None:
            self._customize = MseHttStaticLinearSystemCustomize(self)
        return self._customize

    def pr(self):
        """"""
        if RANK != MASTER_RANK:
            return None

        if self._pr_texts is None:
            print('No texts to print.')
            return None

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
        plt.title(f'Local-Linear-System evaluated @ [{self._str_args}]')
        plt.tight_layout()
        from src.config import _setting, _pr_cache
        if _setting['pr_cache']:
            _pr_cache(fig, filename='msepy_staticLocalLinearSystem')
        else:
            plt.show(block=_setting['block'])

        return fig


class _AAA(Frozen):
    """"""

    def __init__(self, ls, A):
        """"""
        self._ls = ls
        self._mA = bmat(A)
        self._A = A  # save the block-wise A
        self._freeze()

    def __iter__(self):
        """go through all local elements."""
        for i in self._mA:
            yield i

    def __getitem__(self, i):
        """Get the local system for element #i. Adjustments and customizations will take effect."""
        return self._mA[i]

    def spy(self, i, **kwargs):
        """spy plot the local A matrix of element #i."""
        assert (0 <= i < self._mA._gm_row.num_global_elements) and (i % 1 == 0), f"i = {i} is out of range."
        if i in self._mA._gm_row:
            return self._mA.spy(i, **kwargs)
        else:
            return None

    def condition_number(self, i):
        """Return the condition number of local A matrix of element #i in all RANKS."""
        if i in self._mA._gm_row:
            cn = self._mA.condition_number(i)
        else:
            cn = 0
        return COMM.allreduce(cn, MPI.SUM)

    def rank(self, i):
        """Return the rank of local A matrix of element #i in all RANKS."""
        if i in self._mA._gm_row:
            rank = self._mA.rank(i)
        else:
            rank = 0
        return COMM.allreduce(rank, MPI.SUM)

    def num_singularities(self, i):
        """Return the number of singular modes in local A matrix of element #i in all RANKS."""
        if i in self._mA._gm_row:
            ns = self._mA.num_singularities(i)
        else:
            ns = 0
        return COMM.allreduce(ns, MPI.SUM)


class _Xxx(Frozen):
    """"""
    def __init__(self, ls, x):
        """"""
        time_slots = list()
        for c in range(ls.shape[1]):
            assert x[c].__class__ is MseHttStaticCochainVector, f"x[{c}] is not a static cochain vector."
            time_slots.append(
                x[c]._time
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
        """# """
        if isinstance(x, dict):
            self._vx._receive_data(x)
            x_individuals = self._vx.split()
            for i, x_i in enumerate(x_individuals):
                self._x[i]._receive_data(x_i)
                self._x[i].override()

        elif isinstance(x, np.ndarray) and x.ndim == 1:

            gm = self._vx._gm

            _2dx = {}

            for i in gm:
                gmi = gm[i]
                _2dx[i] = x[gmi]

            self._vx._receive_data(_2dx)

            x_individuals = self._vx.split()

            for i, x_i in enumerate(x_individuals):
                self._x[i]._receive_data(x_i)
                self._x[i].override()

        else:
            raise NotImplementedError(f"x.__class__ = {x.__class__} cannot be updated to unknowns.")


class _Bbb(Frozen):
    """"""
    def __init__(self, ls, b):
        """"""
        self._ls = ls
        self._b = b  # save the block-wise b
        self._vb = concatenate(b, ls.A._mA._gm_row)

    def __iter__(self):
        """go through all local elements indices"""
        for i in self._vb:
            yield i

    def __getitem__(self, i):
        """Get the local vector for rank element `#i`."""
        return self._vb[i]
