# -*- coding: utf-8 -*-
r"""
"""
from src.config import RANK, MASTER_RANK

if RANK == MASTER_RANK:
    import matplotlib.pyplot as plt
    import matplotlib
else:
    pass

import numpy as np
from tools.frozen import Frozen
from msehtt.tools.gathering_matrix import MseHttGatheringMatrix
from msehtt.tools.matrix.static.local import MseHttStaticLocalMatrix
from msehtt.tools.vector.static.local import MseHttStaticLocalVector
from msehtt.static.form.addons.static import MseHttFormStaticCopy
from msehtt.static.form.main import MseHttForm

from msehtt.tools.nonlinear_system.static.customize import MseHttStaticNonlinearSystemCustomize
from msehtt.tools.nonlinear_system.static.solve.main import MseHttStaticNonlinearSystemSolve
from msehtt.tools.linear_system.static.local.main import MseHttStaticLocalLinearSystem

from src.wf.mp.linear_system_bc import _EssentialBoundaryCondition


class MseHttStaticNonLinearSystem(Frozen):
    """Must be local-wise (rank-wise)."""

    def __init__(
            self,
            # the linear part
            A, x, b,  # None blocks of ``A``, ``b`` have been replaced by zero matrix.
            _pr_texts=None, _time_indicating_text=None, _str_args='',
            # the nonlinear part
            bc=None,  # since for nonlinear system, not all bcs have taken their effect.
            nonlinear_terms=None,
            nonlinear_signs=None,
            nonlinear_texts=None,
            nonlinear_time_indicators=None,
            test_forms=None,
            unknowns=None,
            configurations=None,
    ):
        # linear part --------------------------------------------------------------------------
        row_shape = len(A)
        col_shape = len(A[0])
        assert len(x) == col_shape and len(b) == row_shape, "A, x, b shape dis-match."
        self._shape = (row_shape, col_shape)
        self._parse_gathering_matrices(A, x, b, nonlinear_terms)
        self._A = A   # A is a 2d list of MseHttStaticLocalMatrix
        self._x = x   # x ia a list of MseHttStaticLocalVector (or subclass)
        self._b = b   # b ia a list of MseHttStaticLocalVector (or subclass)
        self._pr_texts = _pr_texts
        self._time_indicating_text = _time_indicating_text
        self._str_args = _str_args

        # nonlinear part -----------------------------------------------------------------------
        self._bc = bc
        self._nonlinear_terms = nonlinear_terms
        self._nonlinear_signs = nonlinear_signs
        self._nonlinear_texts = nonlinear_texts
        self._nonlinear_time_indicators = nonlinear_time_indicators

        self._configurations = configurations

        self._num_nonlinear_terms = 0
        for i in self._nonlinear_terms:
            self._num_nonlinear_terms += len(self._nonlinear_terms[i])
        for tf in test_forms:
            assert tf._is_base(), f"test forms must be generic or base forms."
        self._tfs = test_forms
        self._uks = unknowns
        self._solve = None
        self._customize = None
        self.___global_row_gm___ = None
        self.___global_col_gm___ = None

        self.___essential_bc_record___ = []
        if self.bc is None:
            pass
        else:
            self.___take_care_of_essential_bc___()
        self._freeze()

    # --------------------- properties ----------------------------------------------------------
    @property
    def bc(self):
        return self._bc

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
        if self._customize is None:
            self._customize = MseHttStaticNonlinearSystemCustomize(self)
        return self._customize

    @property
    def solve(self):
        """"""
        if self._solve is None:
            self._solve = MseHttStaticNonlinearSystemSolve(self)
        return self._solve

    @property
    def linear(self):
        """Make a static local linear system."""
        return MseHttStaticLocalLinearSystem(self._A, self._x, self._b)

    # ------------- gathering matrices -----------------------------------------------------------

    def _parse_gathering_matrices(self, A, x, b, nonlinear_terms):
        """"""
        row_shape, col_shape = self.shape
        row_gms = []
        col_gms = []

        # now we check gathering matrices in x.
        for j in range(col_shape):
            x_j = x[j]
            gm_j = x_j._gm
            col_gms.append(gm_j)

        # now we check gathering matrices in b.
        for i in range(row_shape):
            b_i = b[i]
            if b_i is None:
                row_gms.append(None)
            else:
                assert issubclass(b_i.__class__, MseHttStaticLocalVector), \
                    f"b[{i}] is {b_i.__class__}, wrong"
                gm_i = b_i._gm
                row_gms.append(gm_i)

        assert None not in col_gms, f"miss some gathering matrices in col gms."

        # ------ now we check gathering matrices in A. -------------------------------------
        for i in range(row_shape):
            for j in range(col_shape):
                A_ij = A[i][j]

                if A_ij is None:
                    pass
                else:
                    assert A_ij.__class__ is MseHttStaticLocalMatrix, f"A[{i}][{j}] is {A_ij.__class__}, wrong!"
                    row_gm_i = A_ij._gm_row
                    col_gm_j = A_ij._gm_col

                    assert col_gms[j] is col_gm_j
                    if row_gms[i] is None:
                        row_gms[i] = row_gm_i
                    else:
                        assert row_gms[i] is row_gm_i

        # ------ now we check gathering matrices in nonlinear part. ------------------------
        for i in nonlinear_terms:
            terms = nonlinear_terms[i]
            row_gm_i = row_gms[i]
            for j in range(len(terms)):
                term_ij = terms[j]
                generic_form = None
                for form in term_ij.correspondence:
                    if form.__class__ is MseHttFormStaticCopy:
                        pass
                    elif form.__class__ is MseHttForm:
                        if generic_form is None:
                            generic_form = form
                        else:
                            raise Exception(f"found more than one generic form in the correspondence.")
                    else:
                        raise NotImplementedError()

                if row_gm_i is None:
                    row_gms[i] = generic_form.cochain.gathering_matrix
                else:
                    assert row_gms[i] == generic_form.cochain.gathering_matrix, \
                        f"the rol-gm must match the gm of the test form in a nonlinear term."

        assert None not in row_gms, f"we miss some gm in row-gms."

        # ======================================================================================
        self._row_gms = row_gms
        self._col_gms = col_gms

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

    @property
    def _global_row_gm(self):
        if self.___global_row_gm___ is None:
            self. ___global_row_gm___ = MseHttGatheringMatrix(self._row_gms)
        return self.___global_row_gm___

    @property
    def _global_col_gm_(self):
        if self. ___global_col_gm___ is None:
            self.___global_col_gm___ = MseHttGatheringMatrix(self._col_gms)
        return self.___global_col_gm___

    # -------------- PR part -------------------------------------------------------------------
    def pr(self):
        """"""
        if RANK != MASTER_RANK:
            return None
        else:
            pass

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

        nonlinear_shape_text, nonlinear_time_text = self._pr_nonlinear_text()

        text = tA + tx + nonlinear_shape_text + '=' + tb  # texts showing local shapes

        text = r"$" + text + r"$"
        fig = plt.figure(figsize=(10, 4))
        plt.axis((0, 1, 0, 1))
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

        plt.title(f'Local-Linear-System evaluated @ [{self._str_args}].')

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
                                t = round(t, 3)
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

    # ------------ Essential BC ----------------------------------------------------------------
    def ___take_care_of_essential_bc___(self):
        """"""
        for bc_section_repr in self.bc:
            bcs_on_section = self.bc[bc_section_repr]
            for section_bc in bcs_on_section:
                if section_bc.__class__ is _EssentialBoundaryCondition:
                    the_matching_config = None
                    configurations = self._configurations
                    for config in configurations[::-1]:
                        if config['type'] == 'essential bc':
                            if config['place'].abstract.manifold._sym_repr == bc_section_repr:
                                config_root_form = config['root_form']
                                ith_unknown = section_bc._i
                                unknown = self._x[ith_unknown]
                                f = unknown._f
                                if f._is_base():
                                    pass
                                else:
                                    f = f._base
                                if config_root_form is f:
                                    the_matching_config = config
                                    break
                                else:
                                    pass
                            else:
                                pass
                        else:
                            pass
                    assert the_matching_config is not None, f"must found a matching configuration."
                    # ------------- type ('essential bc', 1) bc --------------------------------------
                    if the_matching_config['category'] == 1:
                        place = the_matching_config['place']
                        condition = the_matching_config['condition']
                        ith_unknown = section_bc._i
                        unknown = self._x[ith_unknown]
                        f = unknown._f
                        time = unknown._time

                        self.___essential_bc_record___.append(
                            (place, condition, time, f, ith_unknown)
                        )

                        global_dofs = place.find_dofs(f, local=False)
                        local_cochain = f.reduce(condition @ time)
                        gm = f.cochain.gathering_matrix
                        global_cochain = gm.assemble(local_cochain, mode='replace')
                        global_cochain = global_cochain[global_dofs]
                        if ith_unknown in self._nonlinear_terms:
                            self.customize.add_customizations_on_hold(
                                'nonlinear_part_essential_bc',
                                {
                                    'ith_unknown': ith_unknown,
                                    'global_dofs': global_dofs,
                                    'global_cochain': global_cochain
                                }
                            )
                        else:
                            # we can apply this essential bc through making changes in the linear part.
                            self.customize.linear.apply_essential_bc_for_unknown(
                                ith_unknown, global_dofs, global_cochain)
                    # ==================================================================================
                    else:
                        raise NotImplementedError()
                else:
                    pass

    # ===========================================================================================

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
                        if isinstance(static_vector, dict):
                            neg_static_vector = dict()
                            for e in static_vector:
                                neg_static_vector[e] = - static_vector[e]
                            static_vector = neg_static_vector
                        else:
                            static_vector = - static_vector

                    if nfi is None:
                        nfi = static_vector

                    else:
                        if isinstance(nfi, dict) or isinstance(static_vector, dict):
                            add_nfi = {}
                            assert len(nfi) == len(static_vector), f"must be"
                            for e in nfi:
                                add_nfi[e] = nfi[e] + static_vector[e]
                            nfi = add_nfi
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

        # -------- nonlinear terms contribution ----------------------------------------
        f_nt = self._evaluate_nonlinear_terms(provided_cochains)
        for i, fi in enumerate(f_nt):
            if fi is None:
                pass
            else:
                if isinstance(fi, dict):
                    fi_2b_added = MseHttStaticLocalVector(
                        fi,
                        self._row_gms[i]
                    )
                elif isinstance(fi, MseHttStaticLocalVector):
                    assert fi._gm == self._row_gms[i], f'must be!'
                    fi_2b_added = fi

                else:
                    raise Exception()

                if f[i] is None:
                    f[i] = fi_2b_added
                else:
                    f[i] += fi_2b_added

        # ------ add the original b to f -----------------------------------------------
        for i in range(S0):
            if f[i] is None:
                f[i] = - self._b[i]   # b[i] cannot be None
            else:
                f[i] += - self._b[i]

        # ------ neg -------------------------------------------------------------------
        if neg:
            for i in range(S0):
                # noinspection PyUnresolvedReferences
                f[i] = - f[i]
        else:
            pass

        # --------- return -------------------------------------------------------------
        return f
