# -*- coding: utf-8 -*-
r"""
"""
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
})
matplotlib.use('TkAgg')
import numpy as np

from tools.frozen import Frozen
from msepy.tools.matrix.static.local import MsePyStaticLocalMatrix
from msepy.tools.vector.static.local import MsePyStaticLocalVector
from msepy.form.cochain.vector.static import MsePyRootFormStaticCochainVector
from msepy.tools.linear_system.static.assembled.main import MsePyStaticLinearSystemAssembled
from msepy.tools.linear_system.static.customize import MsePyStaticLinearSystemCustomize

from msepy.tools.matrix.static.bmat import bmat
from msepy.tools.vector.static.concatenate import concatenate


class MsePyStaticLinearSystem(Frozen):
    """"""

    def __init__(self, A, x, b, _pr_texts=None):
        row_shape = len(A)
        col_shape = len(A[0])
        assert len(x) == col_shape and len(b) == row_shape, "A, x, b shape dis-match."
        self._shape = (row_shape, col_shape)
        self._parse_gathering_matrices(A, x, b)
        self._A = _AAA(self, A)
        self._x = _Xxx(self, x)
        self._b = _Bbb(self, b)
        self._customize = None
        self._pr_texts = _pr_texts
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
                    assert A_ij.__class__ is MsePyStaticLocalMatrix, f"A[{i}][{j}] is {A_ij.__class__}, wrong!"
                    row_gm_i = A_ij._gm0_row
                    col_gm_j = A_ij._gm1_col

                    if row_gms[i] is None:
                        row_gms[i] = row_gm_i
                    else:
                        assert row_gms[i] is row_gm_i, f"by construction, this must be the case as we only construct" \
                                                       f"gathering matrix once and store only once copy somewhere!"

                    if col_gms[j] is None:
                        col_gms[j] = col_gm_j
                    else:
                        assert col_gms[j] is col_gm_j, f"by construction, this must be the case as we only construct" \
                                                       f"gathering matrix once and store only once copy somewhere!"

        assert None not in row_gms and None not in col_gms, f"miss some gathering matrices."
        num_elements = list()
        for rgm in row_gms:
            # noinspection PyUnresolvedReferences
            num_elements.append(rgm.num_elements)
        for cgm in col_gms:
            # noinspection PyUnresolvedReferences
            num_elements.append(cgm.num_elements)
        # noinspection PyTypeChecker
        assert all(np.array(num_elements) == num_elements[0]), f"total element number dis-match"
        self._total_elements = num_elements[0]
        self._row_gms = row_gms
        self._col_gms = col_gms

        # now we check gathering matrices in x.
        for j in range(col_shape):
            x_j = x[j]
            assert x_j.__class__ is MsePyRootFormStaticCochainVector, f"x[{j}] is {x_j.__class__}, wrong"
            gm_j = x_j._gm
            assert gm_j is self._col_gms[j], f"by construction, this must be the case as we only construct" \
                                             f"gathering matrix once and store only once copy somewhere!"

        # now we check gathering matrices in b.
        for i in range(row_shape):
            b_i = b[i]
            if b_i is None:
                pass
            else:
                assert b_i.__class__ in (MsePyStaticLocalVector, MsePyRootFormStaticCochainVector), \
                    f"b[{i}] is {b_i.__class__}, wrong"
                gm_i = b_i._gm
                assert gm_i is self._row_gms[i], f"by construction, this must be the case as we only construct" \
                                                 f"gathering matrix once and store only once copy somewhere!"

    @property
    def gathering_matrices(self):
        """return all gathering matrices; both row and col.

        Note the differences to the `gm0_row` and `gm1_col` of `self.A._A`.
        """
        return self._row_gms, self._col_gms

    @property
    def num_elements(self):
        """total amount of elements."""
        return self._total_elements

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

    def assemble(self):
        """"""
        A = self.A._mA.assemble()
        b = self.b._vb.assemble()
        ALS = MsePyStaticLinearSystemAssembled(self, A, b)
        return ALS

    @property
    def customize(self):
        """customize A and b."""
        if self._customize is None:
            self._customize = MsePyStaticLinearSystemCustomize(self)
        return self._customize

    def pr(self, figsize=(10, 4)):
        """"""
        if self._pr_texts is None:
            print('No texts to print.')
            return
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
        fig = plt.figure(figsize=figsize)
        plt.axis([0, 1, 0, 1])
        plt.axis('off')
        plt.text(0.05, 0.5, text, ha='left', va='center', size=15)
        plt.tight_layout()
        from src.config import _matplot_setting
        plt.show(block=_matplot_setting['block'])

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
        for i in range(self._ls.num_elements):
            yield i

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
            assert x[c].__class__ is MsePyRootFormStaticCochainVector, f"x[{c}] is not a static cochain vector."
            time_slots.append(
                x[c]._time
            )
        self._representing_time = time_slots
        self._ls = ls
        self._x = x  # important!, use to override cochain back to the form.
        self._vx = concatenate(x, ls.A._mA._gm1_col)

    @property
    def representing_time(self):
        """The cochain static vector are at these time instants for the corresponding root-forms."""
        return self._representing_time

    def update(self, x):
        """# """
        gm = self._vx._gm

        _2dx = np.zeros(gm.shape)

        for i in gm:
            gmi = gm[i]
            _2dx[i] = x[gmi]

        self._vx.data = _2dx

        x_individuals = self._vx.split()

        for i, x_i in enumerate(x_individuals):
            self._x[i].data = x_i
            self._x[i].override()


class _Bbb(Frozen):
    """"""
    def __init__(self, ls, b):
        self._ls = ls
        self._b = b  # save the block-wise b
        self._vb = concatenate(b, ls.A._mA._gm0_row)

    def __getitem__(self, i):
        """Get the local vector for element `#i`."""
        return self._vb[i]
