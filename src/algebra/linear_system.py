# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen

import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
})
matplotlib.use('TkAgg')


class BlockMatrix(Frozen):
    """"""
    def __init__(self, shape):
        self._shape = shape
        self._entries = dict()
        self._signs = dict()
        for i in range(shape[0]):
            self._entries[i] = list()
            self._signs[i] = list()
            for _ in range(shape[1]):
                self._entries[i].append(list())
                self._signs[i].append(list())
        self._freeze()

    def _is_empty(self):
        empty = True
        for i in self._entries:
            for en in self._entries[i]:
                if en != list():
                    return False
                else:
                    pass
        return empty

    def _add(self, i, j, term, sign):
        """"""
        assert sign in ('+', '-'), f"sign={sign} is wrong."
        if self._entries[i][j] != list():
            assert term.shape == self._entries[i][j][0].shape, f"shape dis-match."
        else:
            pass
        self._entries[i][j].append(term)
        self._signs[i][j].append(sign)

    def __iter__(self):
        """go through all terms."""
        rows, cols = self._shape
        for i in range(rows):
            for j in range(cols):
                yield i, j

    def __call__(self, i, j):
        """"""
        return self._entries[i][j], self._signs[i][j]

    def _pr_text(self):
        """"""
        symbolic = ''
        for i in self._entries:
            entry = self._entries[i]
            for j, terms in enumerate(entry):
                if len(terms) == 0:
                    symbolic += r"\boldsymbol{0}"

                for k, term in enumerate(terms):
                    sign = self._signs[i][j][k]

                    if k == 0 and sign == '+':
                        symbolic += term._sym_repr

                    else:
                        symbolic += sign + term._sym_repr

                if j < len(entry) - 1:
                    symbolic += '&'

            if i < len(self._entries) - 1:
                symbolic += r'\\'

        symbolic = r"\begin{bmatrix}" + symbolic + r"\end{bmatrix}"

        return symbolic

    def pr(self, figsize=(12, 6)):
        """"""
        from src.config import RANK, MASTER_RANK
        if RANK != MASTER_RANK:
            return
        else:
            pass

        symbolic = r"$" + self._pr_text() + r"$"
        fig = plt.figure(figsize=figsize)
        plt.axis((0, 1, 0, 1))
        plt.axis('off')
        plt.text(0.05, 0.5, symbolic, ha='left', va='center', size=15)
        plt.tight_layout()
        from src.config import _setting
        plt.show(block=_setting['block'])
        return fig


class BlockColVector(Frozen):
    """"""

    def __init__(self, shape):
        """"""
        self._shape = shape
        self._entries = tuple([list() for _ in range(shape)])
        self._signs = tuple([list() for _ in range(shape)])
        self._freeze()

    def __call__(self, i):  # work as getitem, use call to make it consistent with `BlockMatrix`.
        """"""
        return self._entries[i], self._signs[i]

    def __iter__(self):
        """"""
        for i in range(self._shape):
            yield i

    def _is_empty(self):
        empty = True
        for en in self._entries:
            if en != list():
                return False
            else:
                pass
        return empty

    def _add(self, i, term, sign):
        """add to entry #i"""
        assert sign in ('+', '-'), f"sign={sign} is wrong."
        if self._entries[i] != list():
            assert term.shape == self._entries[i][0].shape, f"shape dis-match."
        else:
            pass
        self._entries[i].append(term)
        self._signs[i].append(sign)

    def _pr_text(self):
        """"""
        symbolic = ''
        for i, entry in enumerate(self._entries):

            if len(entry) == 0:
                symbolic += r'\boldsymbol{0}'
            else:
                for j, term in enumerate(entry):
                    sign = self._signs[i][j]

                    if j == 0 and sign == '+':
                        symbolic += term._sym_repr

                    else:
                        symbolic += sign + term._sym_repr

            if i < len(self._entries) - 1:
                symbolic += r'\\'

        symbolic = r"\begin{bmatrix}" + symbolic + r"\end{bmatrix}"
        return symbolic

    def pr(self, figsize=(8, 6)):
        """"""
        from src.config import RANK, MASTER_RANK
        if RANK != MASTER_RANK:
            return
        else:
            symbolic = r"$" + self._pr_text() + r"$"
            fig = plt.figure(figsize=figsize)
            plt.axis((0, 1, 0, 1))
            plt.axis('off')
            plt.text(0.05, 0.5, symbolic, ha='left', va='center', size=15)
            plt.tight_layout()
            from src.config import _setting
            plt.show(block=_setting['block'])
            return fig

    # noinspection PyRedeclaration
    def __iter__(self):
        """iter"""
        for i in range(self._shape):
            yield i


class LinearSystem(Frozen):
    """"""

    def __init__(self, A, x, b):
        """"""
        assert A.__class__ is BlockMatrix, f"A = {A} is not a BlockMatrix."
        assert x.__class__ is BlockColVector, f"x = {x} is not a BlockColVector."
        assert b.__class__ is BlockColVector, f"b = {b} is not a BlockColVector."
        for i in x:
            xi = x(i)
            assert len(xi[0]) == 1, f"x[{i}] = {xi} is wrong."
            # make sure x is not empty and only have one entry on each row.
            entry = xi[0][0]
            assert entry.is_root() and entry.ndim == 2 and entry.shape[1] == 1, \
                f"x[{i}] is illegal, it must be  a root col vector (abstract array of shape (x, 1))."

        A_shape = A._shape
        x_shape = x._shape
        b_shape = b._shape
        assert A_shape[0] == A_shape[1], f"A must be a square block-wise matrix."
        assert A_shape[0] == b_shape, f"A shape dis-match b shape."
        assert A_shape[1] == x_shape, f"A shape dis-match x shape."
        self._b_shape = A_shape
        self._A = A
        self._x = x
        self._b = b

        _shapes = [None for _ in range(b_shape)]

        for i in range(A_shape[0]):
            Bi = b(i)[0]
            for bi in Bi:
                assert bi.ndim == 2 and bi.shape[1] == 1, f"b[{i}] is not a col vector."
                if _shapes[i] is None:
                    _shapes[i] = bi.shape[0]
                else:
                    assert bi.shape[0] == _shapes[i]
            for j in range(A_shape[1]):
                Aij = A(i, j)[0]
                for aij in Aij:
                    if _shapes[i] is None:
                        _shapes[i] = aij.shape[0]
                    else:
                        assert aij.shape[0] == _shapes[i]

        assert None not in _shapes, f"miss row-shape."

        for i in range(A_shape[0]):
            for j in range(A_shape[1]):
                Xj, Sj = x(j)
                assert len(Xj) == 1 and Sj[0] == '+', f"entries of x must be of '+' sign."
                # Aij = A(i, j)[0]
                # xj = Xj[0]
                # for k, aij in enumerate(Aij):
                #     assert aij.shape[1] == _shapes[j], f"{k}th component of A[{i}][{j}], {aij}, shape wrong."
                # assert xj.shape[0] == _shapes[j], f"x[{j}] shape wrong."

        self._shapes = _shapes
        self._freeze()

    @property
    def b_shape(self):
        """block shape of A. For example, if the system is written as
        | M   E  N|| x |   | a |
        | C   D  0|| y | = | b |
        | F   0  0|| z |   | c |
        Then b_shape is (3, 3).

         """
        return self._b_shape

    def shapes(self, i, j=None):
        """How many rows the block matrices have?
        if j == None, we return how many rows the entries of the ith row block have.
        if j is not None, we return the shape of the matrix A[i][j]

        For example, for system,
        | M   E  N|| x |   | a |
        | C   D  0|| y | = | b |
        | F   0  0|| z |   | c |

        shapes(0) gives the row shape of M, E and N.
        shapes(1, 1) gives the shapes of matrix D.

        Note that when we are looking at a None or 0 matrix, it also works.
        """
        if j is None:
            return self._shapes[i]
        else:
            return self._shapes[i], self._shapes[j]

    def _pr_text(self):
        symbolic = ''
        symbolic += self._A._pr_text()
        symbolic += self._x._pr_text()
        symbolic += '='
        symbolic += self._b._pr_text()
        return symbolic

    def pr(self, figsize=(12, 6)):
        """pr"""
        from src.config import RANK, MASTER_RANK
        if RANK != MASTER_RANK:
            return
        else:
            symbolic = r"$" + self._pr_text() + r"$"
            fig = plt.figure(figsize=figsize)
            plt.axis((0, 1, 0, 1))
            plt.axis('off')
            plt.text(0.05, 0.5, symbolic, ha='left', va='center', size=15)
            plt.tight_layout()
            from src.config import _setting
            plt.show(block=_setting['block'])
            return fig

    @property
    def A(self):
        return self._A

    @property
    def x(self):
        return self._x

    @property
    def b(self):
        return self._b
