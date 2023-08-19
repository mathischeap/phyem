# -*- coding: utf-8 -*-
r"""
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from tools.frozen import Frozen
from msepy.tools.matrix.static.local import MsePyStaticLocalMatrix
from msepy.tools.vector.static.local import MsePyStaticLocalVector
from msepy.form.cochain.vector.static import MsePyRootFormStaticCochainVector
from msepy.form.static import MsePyRootFormStaticCopy
from msepy.form.main import MsePyRootForm
from msepy.tools.nonlinear_system.static.customize import MsePyStaticNonlinearSystemCustomize
from msepy.tools.nonlinear_system.static.solve.main import MsePyStaticNonlinearSystemSolve


class MsePyStaticLocalNonLinearSystem(Frozen):
    """"""

    def __init__(
            self,
            # the linear part
            A, x, b,  # None blocks of ``A``, ``b`` have been replaced by zero matrix.
                      # ``x`` is a list of MsePyRootFormStaticCochainVector objects.
            _pr_texts=None, _time_indicating_text=None, _str_args='',
            # the nonlinear part
            bc=None,  # since for nonlinear system, not all bcs have taken their effect.
            nonlinear_terms=None,
            nonlinear_signs=None,
            nonlinear_texts=None,
            nonlinear_time_indicators=None,
            test_forms=None,
            unknowns=None,
    ):
        # linear part
        row_shape = len(A)
        col_shape = len(A[0])
        assert len(x) == col_shape and len(b) == row_shape, "A, x, b shape dis-match."
        self._shape = (row_shape, col_shape)
        self._parse_gathering_matrices(A, x, b, nonlinear_terms)
        self._A = A   # A is a 2d list of MsePyStaticLocalMatrix
        self._x = x   # x ia a list of MsePyStaticLocalVector (or subclass)
        self._b = b   # b ia a list of MsePyStaticLocalVector (or subclass)
        self._pr_texts = _pr_texts
        self._time_indicating_text = _time_indicating_text
        self._str_args = _str_args

        # nonlinear part
        self._bc = bc
        self._nonlinear_terms = nonlinear_terms
        self._nonlinear_signs = nonlinear_signs
        self._nonlinear_texts = nonlinear_texts
        self._nonlinear_time_indicators = nonlinear_time_indicators

        self._num_nonlinear_terms = 0
        for i in self._nonlinear_terms:
            self._num_nonlinear_terms += len(self._nonlinear_terms[i])
        for tf in test_forms:
            assert tf._is_base(), f"test forms must be generic or base forms."
        self._tfs = test_forms
        self._uks = unknowns
        self._solve = MsePyStaticNonlinearSystemSolve(self)
        self._customize = MsePyStaticNonlinearSystemCustomize(self)
        self._freeze()

    @property
    def shape(self):
        """block shape"""
        return self._shape

    @property
    def test_forms(self):
        """generic forms; base forms."""
        return self._tfs

    @property
    def unknowns(self):
        """The unknowns; msepy form copies at particular time instances; not forms."""
        return self._uks

    @property
    def customize(self):
        """"""
        return self._customize

    @property
    def solve(self):
        return self._solve

    def _parse_gathering_matrices(self, A, x, b, nonlinear_terms):
        """"""
        row_shape, col_shape = self.shape
        row_gms = []
        col_gms = []

        # now we check gathering matrices in x.
        for j in range(col_shape):
            x_j = x[j]
            assert x_j.__class__ is MsePyRootFormStaticCochainVector, f"x[{j}] is {x_j.__class__}, wrong"
            gm_j = x_j._gm
            col_gms.append(gm_j)

        # now we check gathering matrices in b.
        for i in range(row_shape):
            b_i = b[i]
            if b_i is None:
                row_gms.append(None)
            else:
                assert b_i.__class__ in (MsePyStaticLocalVector, MsePyRootFormStaticCochainVector), \
                    f"b[{i}] is {b_i.__class__}, wrong"
                gm_i = b_i._gm
                row_gms.append(gm_i)

        assert None not in col_gms, f"miss some gathering matrices in col gms."
        for i in range(row_shape):
            for j in range(col_shape):
                A_ij = A[i][j]

                if A_ij is None:
                    pass
                else:
                    assert A_ij.__class__ is MsePyStaticLocalMatrix, f"A[{i}][{j}] is {A_ij.__class__}, wrong!"
                    row_gm_i = A_ij._gm0_row
                    col_gm_j = A_ij._gm1_col

                    assert col_gms[j] is col_gm_j
                    if row_gms[i] is None:
                        row_gms[i] = row_gm_i
                    else:
                        assert row_gms[i] is row_gm_i

        for i in nonlinear_terms:
            terms = nonlinear_terms[i]
            row_gm_i = row_gms[i]
            for j in range(len(terms)):
                term_ij = terms[j]
                generic_form = None
                for form in term_ij.correspondence:
                    if form.__class__ is MsePyRootFormStaticCopy:
                        pass
                    elif form.__class__ is MsePyRootForm:
                        if generic_form is None:
                            generic_form = form
                        else:
                            raise Exception(f"found more than one generic form in the correspondence.")

                if row_gm_i is None:
                    row_gms[i] = generic_form.cochain.gathering_matrix
                else:
                    assert row_gms[i] == generic_form.cochain.gathering_matrix, \
                        f"the rol-gm must match the gm of the test form in a nonlinear term."

        assert None not in row_gms, f"we miss some gm in row-gms."

        self._row_gms = row_gms
        self._col_gms = col_gms

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

        nonlinear_shape_text, nonlinear_time_text = self._pr_nonlinear_text()

        text = tA + tx + nonlinear_shape_text + '=' + tb  # texts showing local shapes

        text = r"$" + text + r"$"
        fig = plt.figure(figsize=(10, 4))
        plt.axis([0, 1, 0, 1])
        plt.axis('off')
        if self._time_indicating_text is None:
            plt.text(0.05, 0.5, text, ha='left', va='center', size=15)
        else:
            plt.text(0.05, 0.525, text, ha='left', va='bottom', size=15)
            plt.plot([0, 1], [0.5, 0.5], '--', color='gray', linewidth=0.5)

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
            text = tA + tx + nonlinear_time_text + '=' + tb
            text = r"$" + text + r"$"
            plt.text(0.05, 0.47, text, ha='left', va='top', size=15)

        plt.title(f'Local-Linear-System evaluated @ [{self._str_args}]')

        from src.config import _setting, _pr_cache
        if _setting['pr_cache']:
            _pr_cache(fig, filename='msepy_staticLocalNonLinearSystem')
        else:
            plt.tight_layout()
            plt.show(block=_setting['block'])

        return fig

    def _pr_nonlinear_text(self):
        """"""
        nonlinear_shape_text, nonlinear_time_text = '', ''

        nonlinear_texts = self._nonlinear_texts
        nonlinear_signs = self._nonlinear_signs
        nonlinear_times = self._nonlinear_time_indicators

        for i in range(self.shape[0]):
            if i in nonlinear_texts:
                texts = nonlinear_texts[i]
                signs = nonlinear_signs[i]
                times = nonlinear_times[i]
                len_terms = len(texts)
                for j, text in enumerate(texts):
                    sign = signs[j]
                    time = times[j]

                    if j == 0 and sign == '+':
                        final_sign = ''
                    else:
                        final_sign = sign

                    nonlinear_shape_text += final_sign + text

                    if isinstance(time, (int, float)):
                        time = [time, ]
                    else:
                        pass

                    str_time = list()
                    for t in time:
                        if t is None:
                            str_time.append(r'\mathrm{None}')
                        else:
                            t = round(t, 12)
                            if t % 1 == 0:
                                str_time.append(str(int(t)))
                            else:
                                t = round(t, 2)
                                str_time.append(str(t))

                    str_time = r'\left<' + ','.join(str_time) + r'\right>'
                    nonlinear_time_text += str_time

                    if j < len_terms - 1:
                        nonlinear_shape_text += r'&'
                        nonlinear_time_text += r'&'
                    else:
                        pass
            else:
                nonlinear_shape_text += '0'
                nonlinear_time_text += '0'

            if i < self.shape[0] - 1:
                nonlinear_shape_text += r'\\'
                nonlinear_time_text += r'\\'
            else:
                pass

        nonlinear_shape_text = r" + \begin{bmatrix}" + nonlinear_shape_text + r"\end{bmatrix}"
        nonlinear_time_text = r" + \begin{bmatrix}" + nonlinear_time_text + r"\end{bmatrix}"

        return nonlinear_shape_text, nonlinear_time_text

    def _evaluate_nonlinear_terms(self, provided_cochains):
        """"""
        pairs = list()

        for k, uk in enumerate(self.unknowns):
            pairs.append(
                [uk, provided_cochains[k]]
            )

        nonlinear_f = list()
        for i in range(self.shape[0]):
            nfi = None
            if i in self._nonlinear_terms:
                NTs = self._nonlinear_terms[i]
                NSs = self._nonlinear_signs[i]
                for term, sign in zip(NTs, NSs):

                    static_vector = term._evaluate(pairs)

                    if sign == '+':
                        pass
                    else:
                        static_vector = - static_vector

                    if nfi is None:
                        nfi = static_vector
                    else:
                        nfi += static_vector
            else:
                pass

            nonlinear_f.append(nfi)

        return nonlinear_f

    def evaluate_f(self, provided_cochains, neg=False):
        """

        Parameters
        ----------
        provided_cochains :
            must be arranged according to unknowns
        neg

        Returns
        -------

        """
        S0, S1 = self.shape
        f = list()

        # ------ linear terms contribution ---------------------------------------------
        for i in range(S0):
            fi = None
            for j in range(S1):
                Aij = self._A[i][j]
                if Aij is None:
                    pass
                else:
                    fij = Aij @ provided_cochains[j]

                    if fi is None:
                        fi = fij
                    else:
                        fi += fij

            f.append(fi)

        # -------- nonlinear terms contribution --------------------------------------------
        f_nt = self._evaluate_nonlinear_terms(provided_cochains)
        for i, fi in enumerate(f_nt):
            if fi is None:
                pass
            else:
                fi_2b_added = MsePyStaticLocalVector(
                    fi,
                    self._row_gms[i]
                )
                if f[i] is None:
                    f[i] = fi_2b_added
                else:
                    f[i] += fi_2b_added

        # -- add the original b to f --------------------------------
        for i in range(S0):
            if f[i] is None:
                f[i] = - self._b[i]   # b[i] cannot be None
            else:
                f[i] += - self._b[i]

        # ------ return f
        if neg:
            for i in range(S0):
                # noinspection PyUnresolvedReferences
                f[i] = - f[i]
        else:
            pass

        return f
