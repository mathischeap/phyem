# -*- coding: utf-8 -*-
r"""
"""
from typing import List

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from msehy.tools.irregular_gathering_matrix import IrregularGatheringMatrix

from tools.frozen import Frozen
from msehy.tools.matrix.static.local import IrregularStaticLocalMatrix
from msehy.tools.vector.static.local import IrregularStaticLocalVector, IrregularStaticCochainVector
from msehy.tools.linear_system.static.assembled.main import IrregularStaticLinearSystemAssembled
from msehy.tools.linear_system.static.customize import IrregularStaticLinearSystemCustomize

from msehy.tools.matrix.static.bmat import bmat
from msehy.tools.vector.static.concatenate import concatenate


class IrregularStaticLocalLinearSystem(Frozen):
    """"""

    def __init__(
            self,
            A, x, b,
            _pr_texts=None,
            _time_indicating_text=None,
            _gene_indicating_text=None,
            _str_args=''
    ):
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
        self._pr_texts = _pr_texts
        self._time_indicating_text = _time_indicating_text
        self._gene_indicating_text = _gene_indicating_text
        self._str_args = _str_args

        if _gene_indicating_text is None:
            self._g = None
        else:
            self._g = self._parse_generation(_gene_indicating_text)

        self._freeze()

    @staticmethod
    def _parse_generation(linear_generations_indicators):
        """"""
        linear_generations = list()
        A, x, b = linear_generations_indicators
        for i, gen_Ai in enumerate(A):
            for j, gen_Aij in enumerate(gen_Ai):
                for k, _ in enumerate(gen_Aij):
                    assert _ % 1 == 0 and _ >= 0, f'generations A[{i}][{j}][{k}] = {_} is wrong.'
                    linear_generations.append(_)

        for i, gen_xi in enumerate(x):
            for j, gen_xij in enumerate(gen_xi):
                assert gen_xij % 1 == 0 and gen_xij >= 0, f'generations x[{i}][{j}] = {gen_xij} is wrong.'
                linear_generations.append(gen_xij)

        for i, gen_bi in enumerate(b):
            for j, gen_bij in enumerate(gen_bi):
                if isinstance(gen_bij, int):
                    assert gen_bij % 1 == 0 and gen_bij >= 0, f'generations b[{i}][{j}] = {gen_bij} is wrong.'
                    linear_generations.append(gen_bij)
                else:

                    for k, _ in enumerate(gen_bij):
                        assert _ % 1 == 0 and _ >= 0, f'generations b[{i}][{j}][{k}] = {_} is wrong.'
                        linear_generations.append(_)

        assert len(linear_generations) > 0, f"must found a generation."
        g = linear_generations[0]
        assert all([_ == g for _ in linear_generations]), f"linear generation is not consistent!"
        return g

    @property
    def shape(self):
        return self._shape

    def _parse_gathering_matrices(self, A, x, b):
        """"""
        row_shape, col_shape = self.shape
        row_gms: List = [None for _ in range(row_shape)]
        col_gms: List = [None for _ in range(col_shape)]

        for i in range(row_shape):
            for j in range(col_shape):
                A_ij = A[i][j]

                if A_ij is None:
                    pass
                else:
                    assert A_ij.__class__ is IrregularStaticLocalMatrix, f"A[{i}][{j}] is {A_ij.__class__}, wrong!"
                    row_gm_i = A_ij._gm0_row
                    col_gm_j = A_ij._gm1_col

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
        num_cells = list()
        for rgm in row_gms:
            assert rgm.__class__ is IrregularGatheringMatrix, f"gm must be irregular"
            num_cells.append(len(rgm))
        for cgm in col_gms:
            assert cgm.__class__ is IrregularGatheringMatrix, f"gm must be irregular"
            num_cells.append(len(cgm))

        # noinspection PyTypeChecker
        assert all(np.array(num_cells) == num_cells[0]), f"total cell number dis-match"

        self._total_cells = num_cells[0]
        self._row_gms = row_gms
        self._col_gms = col_gms

        # now we check gathering matrices in x.
        for j in range(col_shape):
            x_j = x[j]
            assert x_j.__class__ is IrregularStaticCochainVector, f"x[{j}] is {x_j.__class__}, wrong"
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
                assert b_i.__class__ in (IrregularStaticLocalVector, IrregularStaticCochainVector), \
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
        return self.A._mA._gm0_row, self.A._mA._gm1_col

    @property
    def num_cells(self):
        """total amount of cells."""
        return self._total_cells

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
        ALS = IrregularStaticLinearSystemAssembled(self, A, b, cache=cache)
        return ALS

    @property
    def customize(self):
        """customize A and b."""
        if self._customize is None:
            self._customize = IrregularStaticLinearSystemCustomize(self)
        return self._customize

    def pr(self):
        """"""
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
        num_different_shape_local_systems = None

        for Ai_text in A_text:
            for Aij_text in Ai_text:
                if Aij_text == '':
                    pass
                else:
                    assert isinstance(Aij_text, list), f"We put text in list."
                    for Aij_local_term in Aij_text:
                        assert isinstance(Aij_local_term, dict), f"Aij local term texts are in dict."
                        if num_different_shape_local_systems is None:
                            num_different_shape_local_systems = len(Aij_local_term)
                        else:
                            assert num_different_shape_local_systems == len(Aij_local_term)
        for Xi_text in x_text:
            for Xij_text in Xi_text:
                assert isinstance(Xij_text, dict), f"x_text cannot be empty!"
                if num_different_shape_local_systems is None:
                    num_different_shape_local_systems = len(Xij_text)
                else:
                    assert num_different_shape_local_systems == len(Xij_text)

        for Bi_text in b_text:
            if Bi_text == '':
                pass
            else:
                assert isinstance(Bi_text, list), f"We put text in list."
                for Bij_text in Bi_text:
                    assert isinstance(Bij_text, dict), f"Bij local term texts are in dict."
                    if num_different_shape_local_systems is None:
                        num_different_shape_local_systems = len(Bij_text)
                    else:
                        assert num_different_shape_local_systems == len(Bij_text)

        _shape_texts = list()

        for k in range(num_different_shape_local_systems):  # we will make this many text plots.
            tA = ''
            tx = ''
            tb = ''
            for i in range(I_):
                for j in range(_J):
                    tA_ij = A_text[i][j]
                    if tA_ij == '':
                        tA_ij = '0'
                    else:
                        local_text = ''
                        for local_term in tA_ij:
                            local_text += local_term[k]
                        tA_ij = local_text
                    tA += tA_ij
                    if j < _J - 1:
                        tA += '&'
                if i < I_ - 1:
                    tA += r'\\'
            tA = r"\begin{bmatrix}" + tA + r"\end{bmatrix}"
            for j in range(_J):
                tx_j = x_text[j]
                assert isinstance(tx_j, list), f'must be'
                local_text = ''
                for local_term in tx_j:
                    local_text += local_term[k]
                tx += local_text
                if j < _J - 1:
                    tx += r'\\'
            tx = r"\begin{bmatrix}" + tx + r"\end{bmatrix}"
            for i in range(I_):
                tb_i = b_text[i]
                if tb_i == '':
                    tb_i = '0'
                else:
                    local_text = ''
                    for local_term in tb_i:
                        local_text += local_term[k]
                    tb_i = local_text
                tb += tb_i
                if i < I_ - 1:
                    tb += r'\\'
            tb = r"\begin{bmatrix}" + tb + r"\end{bmatrix}"
            text = tA + tx + '=' + tb  # texts showing local shapes
            text = r"$" + text + r"$"
            _shape_texts.append(text)

        for k, text in enumerate(_shape_texts):
            fig = plt.figure(figsize=(10, 4))
            plt.axis([0, 1, 0, 1])
            plt.axis('off')
            if self._time_indicating_text is None:
                plt.text(0.05, 0.5, text, ha='left', va='center', size=15)
            else:
                plt.text(0.05, 0.525, text, ha='left', va='bottom', size=15)
                plt.plot([0, 1], [0.5, 0.5], '--', color='gray', linewidth=0.5)
                At_text, xt_text, bt_text = self._time_indicating_text
                Ag_text, xg_text, bg_text = self._gene_indicating_text
                tA = ''
                tx = ''
                tb = ''
                for i in range(I_):
                    for j in range(_J):
                        tA_ij = At_text[i][j]
                        if tA_ij == '':
                            tA_ij = r'\divideontimes'
                        else:
                            pass
                        g_text = Ag_text[i][j]
                        if len(g_text) == 0:
                            g_text = r'\circledast'
                        else:
                            g_text = str(g_text[0])
                        tA += r'\lceil' + tA_ij + r'\Subset' + g_text + r'\rfloor'
                        if j < _J - 1:
                            tA += '&'
                    if i < I_ - 1:
                        tA += r'\\'
                tA = r"\begin{bmatrix}" + tA + r"\end{bmatrix}"
                for j in range(_J):
                    tx_j = xt_text[j]
                    if tx_j == '':
                        tx_j = r'\divideontimes'

                    g_text = xg_text[j]
                    g_text = str(g_text[0])
                    tx += r'\lceil' + tx_j + r'\Subset' + g_text + r'\rfloor'

                    if j < _J - 1:
                        tx += r'\\'
                tx = r"\begin{bmatrix}" + tx + r"\end{bmatrix}"
                for i in range(I_):
                    tb_i = bt_text[i]
                    if tb_i == '':
                        tb_i = r'\divideontimes'

                    g_text = bg_text[i]
                    if len(g_text) == 0:
                        g_text = r'\circledast'
                    else:
                        g_text = str(g_text[0])

                    tb += r'\lceil' + tb_i + r'\Subset' + g_text + r'\rfloor'
                    if i < I_ - 1:
                        tb += r'\\'
                tb = r"\begin{bmatrix}" + tb + r"\end{bmatrix}"
                text = tA + tx + '=' + tb
                text = r"$" + text + r"$"
                plt.text(0.05, 0.47, text, ha='left', va='top', size=15)
            plt.title(f'\#{k} Local-Linear-System evaluated @ [{self._str_args}]')
            plt.tight_layout()
            from src.config import _setting, _pr_cache
            if _setting['pr_cache']:
                _pr_cache(fig, filename='msepy_staticLocalLinearSystem')
            else:
                plt.show(block=_setting['block'])


class _AAA(Frozen):
    """"""

    def __init__(self, ls, A):
        self._ls = ls
        self._mA = bmat(A)
        self._A = A  # save the block-wise A
        self._freeze()

    def __iter__(self):
        """go through all local cells."""
        for i in range(self._ls.num_cells):
            yield i

    def __getitem__(self, i):
        """Get the local system for cell #i. Adjustments and customizations will take effect."""
        return self._mA[i]

    def spy(self, i, **kwargs):
        """spy plot the local A matrix of cell #i."""
        return self._mA.spy(i, **kwargs)


class _Xxx(Frozen):
    """"""
    def __init__(self, ls, x):
        time_slots = list()
        for c in range(ls.shape[1]):
            assert x[c].__class__ is IrregularStaticCochainVector, f"x[{c}] is not a static cochain vector."
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
        print(111)
        gm = self._vx._gm

        data_dict = dict()

        for i in gm:
            gmi = gm[i]
            data_dict[i] = x[gmi]

        self._vx._set_data(data_dict)

        x_individuals = self._vx.split()

        for i, x_i in enumerate(x_individuals):
            self._x[i]._set_data(x_i)
            self._x[i].override()


class _Bbb(Frozen):
    """"""
    def __init__(self, ls, b):
        self._ls = ls
        self._b = b  # save the block-wise b
        self._vb = concatenate(b, ls.A._mA._gm0_row)

    def __getitem__(self, i):
        """Get the local vector for cell `#i`."""
        return self._vb[i]
