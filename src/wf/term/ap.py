# -*- coding: utf-8 -*-
r"""
"""
from src.config import _wf_term_default_simple_patterns as _simple_patterns
from src.spaces.ap import *
from src.form.parameters import ConstantScalar0Form, constant_scalar
from src.config import _parse_lin_repr  # _parse_lin_repr('TermNonLinearAlgebraicProxy', lin_repr)
from src.config import _nonlinear_ap_test_form_repr
from src.algebra.nonlinear_operator import AbstractNonlinearOperator

from tools.frozen import Frozen
_cs1 = constant_scalar(1)


class TermNonLinearOperatorAlgebraicProxy(Frozen):
    """It is basically a wrapper of an abstract multidimensional array paired with abstract forms
    for each dimension.
    """
    def __init__(self, abstract_multidimensional_array, correspondence):
        """

        Parameters
        ----------
        abstract_multidimensional_array
        correspondence
        """
        assert abstract_multidimensional_array.__class__ is AbstractNonlinearOperator, \
            f"I need a {AbstractNonlinearOperator}."
        self._ama = abstract_multidimensional_array
        self._correspondence = correspondence
        self._tf = None
        self._freeze()

    def __repr__(self):
        """repr"""
        super_repr = super().__repr__().split('object')[1]
        return rf"<nonlinear term ap {self._sym_repr}" + super_repr

    @property
    def _sym_repr(self):
        if self._tf is None:
            if self.factor == _cs1:
                return self._ama._sym_repr
            else:
                return self.factor._sym_repr + self._ama._sym_repr
        else:
            if self.factor == _cs1:
                return self._ama._sym_repr + _nonlinear_ap_test_form_repr['sym'] + self._tf._sym_repr
            else:
                return (self.factor._sym_repr + self._ama._sym_repr +
                        _nonlinear_ap_test_form_repr['sym'] + self._tf._sym_repr)

    @property
    def factor(self):
        return self._ama._factor

    @property
    def _lin_repr(self):
        if self._tf is None:
            pure_lin_repr = self._ama._pure_lin_repr
        else:
            pure_lin_repr = self._ama._pure_lin_repr + _nonlinear_ap_test_form_repr['lin'] + self._tf._pure_lin_repr
        return _parse_lin_repr('TermNonLinearAlgebraicProxy', pure_lin_repr)[0]

    @property
    def _pure_lin_repr(self):
        if self._tf is None:
            pure_lin_repr = self._ama._pure_lin_repr
        else:
            pure_lin_repr = self._ama._pure_lin_repr + _nonlinear_ap_test_form_repr['lin'] + self._tf._pure_lin_repr
        return _parse_lin_repr('TermNonLinearAlgebraicProxy', pure_lin_repr)[1]

    def set_test_form(self, tf):
        """"""
        assert self._tf is None, f"change test form of nonlinear term is dangerous."
        assert tf in self._correspondence, \
            f"tf is not in the corresponding axis forms of the abstract_multidimensional_array"
        self._tf = tf

    # noinspection PyAugmentAssignment
    def __rmul__(self, other):
        """"""
        self._ama = other * self._ama
        return self

    @property
    def correspondence(self):
        """"""
        return self._correspondence


class TermLinearAlgebraicProxy(Frozen):
    """It is basically a wrapper of an abstract array of shape (1, 1). Since it is for
    a weak-formulation term, so its shape must be (1,1)
    ."""

    def __init__(self, abstract_array):
        """"""
        assert abstract_array.shape == (1, 1), f"term shape = {abstract_array.shape} wrong."
        self._sym_repr = abstract_array._sym_repr
        self._lin_repr = abstract_array._lin_repr
        self._abstract_array = abstract_array
        self._freeze()

    def __repr__(self):
        """repr"""
        super_repr = super().__repr__().split('object')[1]
        return f"<TermLinearAlgebraicProxy {self._sym_repr}" + super_repr

    def __rmul__(self, other):
        """other * self"""
        if isinstance(other, (int, float)) or other.__class__ is ConstantScalar0Form:
            return self.__class__(other * self._abstract_array)
        else:
            raise NotImplementedError()

    @staticmethod
    def _is_linear():
        """This is a linear term."""
        return True


class _SimplePatternAPParser(Frozen):
    """"""

    def __init__(self, wft):
        """"""
        self._wft = wft
        self._freeze()

    def __call__(self, test_form=None):
        """"""
        sp = self._wft._simple_pattern
        if sp == '':
            raise NotImplementedError(f"We do not have an pattern for term {self._wft}.")
        else:   # factor is `ConstantScalar0Form`
            if sp == _simple_patterns['(pt,)']:
                return self._parse_reprs_pt(test_form=test_form)
            elif sp == _simple_patterns['(rt,rt)']:
                return self._parse_reprs_rt_rt(test_form=test_form)
            elif sp == _simple_patterns['<|>']:
                return self._parse_reprs_dp(test_form=test_form)
            elif sp == _simple_patterns['(d,)']:
                return self._parse_reprs_d_(test_form=test_form)
            elif sp == _simple_patterns['(,d)']:
                return self._parse_reprs__d(test_form=test_form)

            elif sp == _simple_patterns['(d,d)']:
                return self._parse_reprs_dd(test_form=test_form)

            elif sp == _simple_patterns['(trB, A)'] or sp == _simple_patterns['(A, trB)']:
                return self._parse_reprs_A_trB(test_form=test_form)

            elif sp == _simple_patterns['<d,>']:
                return self._parse_reprs_d_dual_(test_form=test_form)
            elif sp == _simple_patterns['<,d>']:
                return self._parse_reprs__dual_d(test_form=test_form)

            elif sp == _simple_patterns['<tr star | tr >']:
                return self._parse_reprs_tr_star_star(test_form=test_form)

            elif sp == _simple_patterns['(*x*,)']:
                return self._parse_reprs_astA_x_astB_ip_C(test_form=test_form)
            elif sp == _simple_patterns['(*x,)']:
                return self._parse_reprs_astA_x_B_ip_C(test_form=test_form)
            elif sp == _simple_patterns['(x*,)']:
                return self._parse_reprs_A_x_astB_ip_C(test_form=test_form)
            elif sp == _simple_patterns['(x,)']:  # nonlinear term A, B, C all are unknown
                return self._parse_reprs_A_x_B_ip_C(test_form)

            elif sp == _simple_patterns['(* .V *, C)']:
                return self._parse_reprs_astA_convect_astB_ip_C(test_form=test_form)

            elif sp == _simple_patterns['<*x*|C>']:
                return self._parse_reprs_astA_x_astB_dp_C(test_form=test_form)
            elif sp == _simple_patterns['<Ax*|C>']:
                return self._parse_reprs_A_x_astB_dp_C(test_form=test_form)
            elif sp == _simple_patterns['<*xB|C>']:
                return self._parse_reprs_astA_x_B_dp_C(test_form=test_form)
            elif sp == _simple_patterns['<AxB|C>']:
                return self._parse_reprs_A_x_B_dp_C(test_form=test_form)

            # --------- (A, BC) --------------------------------------------------------
            elif sp == _simple_patterns['(A, BC)']:  # nonlinear
                return self._parse_reprs_A_ip_BC(test_form)
            elif sp == _simple_patterns['(*, BC)']:  # linear, matrix, A given
                return self._parse_reprs_astA_ip_B_C(test_form=test_form)
            elif sp == _simple_patterns['(A, *C)']:  # linear, matrix, B given
                return self._parse_reprs_A_ip_astB_C(test_form=test_form)
            elif sp == _simple_patterns['(*, *C)']:  # linear, vector, A and B given
                return self._parse_reprs_astA_ip_astB_C(test_form=test_form)
            # ==========================================================================

            # --------- <AB|C> --------------------------------------------------------
            elif sp == _simple_patterns['<AB|C>']:  # nonlinear
                return self._parse_reprs_AB_dp_C(test_form)
            elif sp == _simple_patterns['<*B|C>']:  # linear, matrix, A given
                return self._parse_reprs_astA_B_dp_C(test_form=test_form)
            elif sp == _simple_patterns['<A*|C>']:  # linear, matrix, B given
                return self._parse_reprs_A_astB_dp_C(test_form=test_form)
            elif sp == _simple_patterns['<**|C>']:  # linear, vector, A and B given
                return self._parse_reprs_astA_astB_dp_C(test_form=test_form)
            # ==========================================================================

            # --------- (AB, C) --------------------------------------------------------
            elif sp == _simple_patterns['(AB, C)']:  # nonlinear
                return self._parse_reprs_AB_ip_C(test_form)
            elif sp == _simple_patterns['(*B, C)']:  # linear, matrix, A given
                return self._parse_reprs_astA_B_ip_C(test_form=test_form)
            elif sp == _simple_patterns['(A*, C)']:  # linear, matrix, B given
                return self._parse_reprs_A_astB_ip_C(test_form=test_form)
            elif sp == _simple_patterns['(**, C)']:  # linear, vector, A and B given
                return self._parse_reprs_astA_astB_ip_C(test_form=test_form)
            # ==========================================================================

            # --------- (AB, d(C)) -----------------------------------------------------
            elif sp == _simple_patterns['(AB, d(C))']:  # nonlinear
                return self._parse_reprs_AB_ip_dC(test_form)
            elif sp == _simple_patterns['(*B, d(C))']:  # linear, matrix, A given
                return self._parse_reprs_astA_B_ip_dC(test_form=test_form)
            elif sp == _simple_patterns['(A*, d(C))']:  # linear, matrix, B given
                return self._parse_reprs_A_astB_ip_dC(test_form=test_form)
            elif sp == _simple_patterns['(**, d(C))']:  # linear, vector, A and B given
                return self._parse_reprs_astA_astB_ip_dC(test_form=test_form)
            # ==========================================================================

            # --------- <AB|d(C)> -----------------------------------------------------
            elif sp == _simple_patterns['<AB|d(C)>']:  # nonlinear
                return self._parse_reprs_AB_dp_dC(test_form)
            elif sp == _simple_patterns['<*B|d(C)>']:  # linear, matrix, A given
                return self._parse_reprs_astA_B_dp_dC(test_form=test_form)
            elif sp == _simple_patterns['<A*|d(C)>']:  # linear, matrix, B given
                return self._parse_reprs_A_astB_dp_dC(test_form=test_form)
            elif sp == _simple_patterns['<**|d(C)>']:  # linear, vector, A and B given
                return self._parse_reprs_astA_astB_dp_dC(test_form=test_form)
            # ==========================================================================

            elif sp == _simple_patterns['<*x*|d(C)>']:
                return self._parse_reprs_astA_x_astB_dp_dC(test_form=test_form)

            elif sp == _simple_patterns['(*x*,*x)']:
                return self._parse_reprs_astA_x_astB__ip__astC_x_D(test_form=test_form)
            elif sp == _simple_patterns['(x*,*x)']:
                return self._parse_reprs_A_x_astB__ip__astC_x_D(test_form=test_form)

            elif sp == _simple_patterns['<*x*|*xD>']:
                return self._parse_reprs_astA_x_astB__dp__astC_x_D(test_form=test_form)

            elif sp == _simple_patterns['(*x,d)']:
                return self._parse_reprs_astA_x_B_ip_dC(test_form=test_form)
            elif sp == _simple_patterns['(x*,d)']:
                return self._parse_reprs_A_x_astB_ip_dC(test_form=test_form)
            elif sp == _simple_patterns['(*x*,d)']:
                return self._parse_reprs_astA_x_astB_ip_dC(test_form=test_form)
            elif sp == _simple_patterns['(AxB,dC)']:
                return self._parse_reprs_A_x_B_ip_dC(test_form=test_form)  # nonlinear pattern

            elif sp == _simple_patterns['(d0*,0*tp)']:  # vector
                return self._parse_reprs_dastA_astA_tp_C(test_form=test_form)  #
            elif sp == _simple_patterns['(d0*,tp0*)']:  # vector
                return self._parse_reprs_dastA_tB_tp_astA(test_form=test_form)  #
            elif sp == _simple_patterns['(d,0*tp0*)']:  # vector
                return self._parse_reprs_dA_astB_tp_astB(test_form=test_form)
            elif sp == _simple_patterns['(d,tp):1K']:  # matrix
                return self._parse_reprs_dA_B_tp_C__1Known(test_form=test_form)  #
            elif sp == _simple_patterns['(d,tp):2K']:  # matrix
                return self._parse_reprs_dA_B_tp_C__2Known(test_form=test_form)  #
            elif sp == _simple_patterns['(d,tp)']:  # nonlinear term, A, B, C all are unknown
                return self._parse_reprs_dA_B_tp_C(test_form=test_form)

            elif sp == _simple_patterns['(,tp):1K']:  # matrix
                return self._parse_reprs_A_B_tp_C__1Known(test_form=test_form)  #
            elif sp == _simple_patterns['(,tp):2K']:  # matrix
                return self._parse_reprs_A_B_tp_C__2Known(test_form=test_form)  #
            elif sp == _simple_patterns['(,tp)']:  # nonlinear term, A, B, C all are unknown
                return self._parse_reprs_A_B_tp_C(test_form=test_form)

            elif sp == _simple_patterns['(<db>,d<b>)']:
                return self._parse_reprs_dbdb(test_form=test_form)

            elif sp == _simple_patterns['(,d-pi)']:
                return self._parse_reprs___d_pi(test_form=test_form)

            elif sp == _simple_patterns['(d(*A),B)']:
                return self._parse_reprs_d_star_A_ip_B(test_form)

            else:
                raise NotImplementedError(f"not implemented for pattern = {sp}")

    def _parse_reprs_pt(self, test_form=None):
        """"""
        spk = self._wft.___simple_pattern_keys___
        bf0 = spk['rsf0']
        d0 = bf0._degree
        s0 = self._wft._f0.space
        s1 = self._wft._f1.space
        d1 = self._wft._f1._degree
        mass_matrix = _VarPar_M(s0, s1, d0, d1)
        v1 = self._wft._f1.ap()

        if test_form == self._wft._f1:
            v0 = bf0.ap()
            pv0 = v0._partial_t()
            term_ap = v1.T @ mass_matrix @ pv0
        else:
            v0 = bf0.ap().T
            pv0 = v0._partial_t()
            term_ap = pv0 @ mass_matrix @ v1
        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)

        sign = '+'
        return term, sign, 'linear'

    def _parse_reprs_rt_rt(self, test_form=None):
        """"""
        f0, f1 = self._wft._f0, self._wft._f1
        s0 = f0.space
        s1 = f1.space
        d0 = f0._degree
        d1 = f1._degree
        mass_matrix = _VarPar_M(s0, s1, d0, d1)
        if test_form == f1:
            v0 = f0.ap()
            v1 = f1.ap().T
            term_ap = v1 @ mass_matrix @ v0
        else:
            v0 = f0.ap().T
            v1 = f1.ap()
            term_ap = v0 @ mass_matrix @ v1
        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'
        return term, sign, 'linear'

    def _parse_reprs_dp(self, test_form):
        """"""
        f0, f1 = self._wft._f0, self._wft._f1
        if test_form == f0:
            A = f0
            B = f1
        elif test_form == f1:
            A = f1
            B = f0
        else:
            raise Exception()

        W_matrix = _VarPar_dp(A, B)  # <A|B>
        vA = A.ap().T
        vB = B.ap()
        term_ap = vA @ W_matrix @ vB
        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'
        return term, sign, 'linear'

    def _parse_reprs_d_(self, test_form=None):
        """"""
        spk = self._wft.___simple_pattern_keys___
        bf0 = spk['rsf0']
        d0 = bf0._degree
        s0 = self._wft._f0.space
        s1 = self._wft._f1.space
        d1 = self._wft._f1._degree
        mass_matrix = _VarPar_M(s0, s1, d0, d1)

        if test_form == self._wft._f1:
            d_matrix = _VarPar_E(bf0)
            v0 = bf0.ap()
            v1 = self._wft._f1.ap().T
            term_ap = v1 @ mass_matrix @ d_matrix @ v0
        else:
            dT_matrix = _VarPar_E(bf0, transpose=True)
            v0 = bf0.ap().T
            v1 = self._wft._f1.ap()
            term_ap = v0 @ dT_matrix @ mass_matrix @ v1
        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'
        return term, sign, 'linear'

    def _parse_reprs_d_dual_(self, test_form=None):
        spk = self._wft.___simple_pattern_keys___
        bf0 = spk['rsf0']
        if test_form == self._wft._f1:
            d_matrix = _VarPar_E(bf0)
            v0 = bf0.ap()
            v1 = self._wft._f1.ap().T
            term_ap = v1 @ d_matrix @ v0
        else:
            dT_matrix = _VarPar_E(bf0, transpose=True)
            v0 = bf0.ap().T
            v1 = self._wft._f1.ap()
            term_ap = v0 @ dT_matrix @ v1
        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'
        return term, sign, 'linear'

    def _parse_reprs__d(self, test_form=None):
        """"""
        s0 = self._wft._f0.space
        d0 = self._wft._f0._degree

        spk = self._wft.___simple_pattern_keys___
        bf1 = spk['rsf1']
        d1 = bf1._degree
        s1 = self._wft._f1.space
        mass_matrix = _VarPar_M(s0, s1, d0, d1)

        if test_form == bf1:
            dT_matrix = _VarPar_E(bf1, transpose=True)
            v0 = self._wft._f0.ap()
            v1 = bf1.ap().T
            term_ap = v1 @ dT_matrix @ mass_matrix @ v0
        else:
            d_matrix = _VarPar_E(bf1)
            v0 = self._wft._f0.ap().T
            v1 = bf1.ap()
            term_ap = v0 @ mass_matrix @ d_matrix @ v1
        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'
        return term, sign, 'linear'

    def _parse_reprs_dd(self, test_form):
        """"""
        spk = self._wft.___simple_pattern_keys___
        # (d bf0, d bf1)
        bf0 = spk['rsf0']
        bf1 = spk['rsf1']

        s0 = self._wft._f0.space  # space of d bf0
        s1 = self._wft._f1.space  # space of d bf1
        d0 = bf0._degree   # degree of d bf0, same to that of bf0
        d1 = bf1._degree   # degree of d bf1, same to that of bf1

        if test_form == bf1:
            mass_matrix = _VarPar_M(s1, s0, d1, d0)

            dT_matrix = _VarPar_E(bf1, transpose=True)
            d_matrix = _VarPar_E(bf0)
            v0 = bf0.ap()
            v1 = bf1.ap().T
            term_ap = v1 @ dT_matrix @ mass_matrix @ d_matrix @ v0

        elif test_form == bf0:
            mass_matrix = _VarPar_M(s0, s1, d0, d1)

            dT_matrix = _VarPar_E(bf0, transpose=True)
            d_matrix = _VarPar_E(bf1)
            v0 = bf1.ap()
            v1 = bf0.ap().T
            term_ap = v1 @ dT_matrix @ mass_matrix @ d_matrix @ v0

        else:
            raise Exception

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'
        return term, sign, 'linear'

    def _parse_reprs__dual_d(self, test_form=None):
        """"""
        spk = self._wft.___simple_pattern_keys___
        bf1 = spk['rsf1']

        if test_form == bf1:
            dT_matrix = _VarPar_E(bf1, transpose=True)
            v0 = self._wft._f0.ap()
            v1 = bf1.ap().T
            term_ap = v1 @ dT_matrix @ v0
        else:
            d_matrix = _VarPar_E(bf1)
            v0 = self._wft._f0.ap().T
            v1 = bf1.ap()
            term_ap = v0 @ d_matrix @ v1
        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'
        return term, sign, 'linear'

    def _parse_reprs_tr_star_star(self, test_form=None):
        """<tr star bf0, tr bf1>"""
        spk = self._wft.___simple_pattern_keys___
        bf0 = spk['rsf0']
        bf1 = spk['rsf1']

        v1 = bf1.ap()
        # s1 = bf1.space
        boundary_wedge_vector = _VarPar_boundary_dp_vector(bf0, bf1)

        if test_form == bf1:  # when bf1 is the testing form.
            term_ap = v1.T @ boundary_wedge_vector
        else:
            raise NotImplementedError()

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'
        return term, sign, 'linear'

    # -------- (A, trB) -------------------------------------------------
    def _parse_reprs_A_trB(self, test_form):
        """"""
        f0, f1 = self._wft._f0, self._wft._f1

        spk = self._wft.___simple_pattern_keys___
        A = spk['A']   # (A, tr B)
        B = spk['B']   # (A, tr B)

        M_space = A.space
        assert M_space is f0.space and M_space is f1.space

        dA = A._degree
        dB = B._degree

        mass_matrix = _VarPar_M(M_space, M_space, dA, dB)   # shape -> [dA, dB]

        trace_matrix = _VarPar_tM(B)

        vA = A.ap()
        vB = B.ap()

        if test_form == A:

            term_ap = vA.T @ mass_matrix @ trace_matrix @ vB

        elif test_form == B:

            term_ap = vB.T @ trace_matrix.T @ mass_matrix @ vA

        else:
            raise Exception()

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'

        return term, sign, 'linear'

    # --- (A .V B, C) ----------------------------------------------------------------------
    def _parse_reprs_astA_convect_astB_ip_C(self, test_form):
        """(*A .V *B, C)"""
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['A'], spk['B'], spk['C']

        if test_form == C:

            cpm = _VarPar_astA_convect_astB_ip_tC(A, B, C)  # a root-array matrix

            v0 = C.ap().T
            term_ap = v0 @ cpm

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'

        return term, sign, 'linear'

    # -------- (A, BC) ----------------------------------------------------------------------
    def _parse_reprs_A_ip_BC(self, test_form):
        r""" Nonlinear, A, B and C are all unknown.

        Returns
        -------

        """
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['A'], spk['B'], spk['C']

        assert test_form in (A, B, C)

        multi_dimensional_array = _VarPar_A_ip_BC(A, B, C)

        term = self._wft._factor * TermNonLinearOperatorAlgebraicProxy(
            multi_dimensional_array,
            [A, B, C]
        )
        sign = '+'

        term.set_test_form(test_form)

        return term, sign, 'nonlinear'

    def _parse_reprs_astA_ip_B_C(self, test_form=None):
        r"""(A, BC)

        A is known.
        """
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['A'], spk['B'], spk['C']

        if test_form == C:

            cpm = _VarPar_astA_ip_B_tC(A, B, C)  # a root-array matrix

            v0 = C.ap().T
            v1 = B.ap()
            term_ap = v0 @ cpm @ v1

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'

        return term, sign, 'linear'

    def _parse_reprs_A_ip_astB_C(self, test_form=None):
        r"""(A, BC)

        B is known.
        """
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['A'], spk['B'], spk['C']

        if test_form == C:

            cpm = _VarPar_A_ip_astB_tC(A, B, C)  # a root-array matrix

            v0 = C.ap().T
            v1 = A.ap()
            term_ap = v0 @ cpm @ v1

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'

        return term, sign, 'linear'

    def _parse_reprs_astA_ip_astB_C(self, test_form=None):
        r"""(A, BC)

        A and B are known.
        """

        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['A'], spk['B'], spk['C']

        if test_form == C:

            cpm = _VarPar_astA_ip_astB_tC(A, B, C)  # a root-array matrix

            v0 = C.ap().T
            term_ap = v0 @ cpm

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'
        return term, sign, 'linear'

    # ============== (A, BC) =====================================================================

    # -------- <AB|C> ----------------------------------------------------------------------------
    def _parse_reprs_AB_dp_C(self, test_form):
        r""" Nonlinear, A, B and C are all unknown.

        Returns
        -------

        """
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['A'], spk['B'], spk['C']

        assert test_form in (A, B, C)

        multi_dimensional_array = _VarPar_AB_dp_C(A, B, C)

        term = self._wft._factor * TermNonLinearOperatorAlgebraicProxy(
            multi_dimensional_array,
            [A, B, C]
        )
        sign = '+'

        term.set_test_form(test_form)

        return term, sign, 'nonlinear'

    def _parse_reprs_astA_B_dp_C(self, test_form=None):
        r"""<AB|C>

        A is known.
        """
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['A'], spk['B'], spk['C']

        if test_form == C:

            cpm = _VarPar_astA_B_dp_tC(A, B, C)  # a root-array matrix

            v0 = C.ap().T
            v1 = B.ap()
            term_ap = v0 @ cpm @ v1

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'

        return term, sign, 'linear'

    def _parse_reprs_A_astB_dp_C(self, test_form=None):
        r"""<AB|C>

        B is known.
        """
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['A'], spk['B'], spk['C']

        if test_form == C:

            cpm = _VarPar_A_astB_dp_tC(A, B, C)  # a root-array matrix

            v0 = C.ap().T
            v1 = A.ap()
            term_ap = v0 @ cpm @ v1

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'

        return term, sign, 'linear'

    def _parse_reprs_astA_astB_dp_C(self, test_form=None):
        r"""<AB|C>

        A and B are known.
        """

        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['A'], spk['B'], spk['C']

        if test_form == C:

            cpm = _VarPar_astA_astB_dp_tC(A, B, C)  # a root-array matrix

            v0 = C.ap().T
            term_ap = v0 @ cpm

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'
        return term, sign, 'linear'

    # ============== <AB|C> =====================================================================

    # -------- (AB, C) --------------------------------------------------------------------------
    def _parse_reprs_AB_ip_C(self, test_form):
        r""" Nonlinear, A, B and C are all unknown.

        Returns
        -------

        """
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['A'], spk['B'], spk['C']

        assert test_form in (A, B, C)

        multi_dimensional_array = _VarPar_AB_ip_C(A, B, C)

        term = self._wft._factor * TermNonLinearOperatorAlgebraicProxy(
            multi_dimensional_array,
            [A, B, C]
        )
        sign = '+'

        term.set_test_form(test_form)

        return term, sign, 'nonlinear'

    def _parse_reprs_astA_B_ip_C(self, test_form=None):
        r"""(AB, C)

        A is known.
        """
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['A'], spk['B'], spk['C']

        if test_form == C:

            cpm = _VarPar_astA_B_ip_tC(A, B, C)  # a root-array matrix

            v0 = C.ap().T
            v1 = B.ap()
            term_ap = v0 @ cpm @ v1

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'

        return term, sign, 'linear'

    def _parse_reprs_A_astB_ip_C(self, test_form=None):
        r"""(AB, C)

        B is known.
        """
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['A'], spk['B'], spk['C']

        if test_form == C:

            cpm = _VarPar_A_astB_ip_tC(A, B, C)  # a root-array matrix

            v0 = C.ap().T
            v1 = A.ap()
            term_ap = v0 @ cpm @ v1

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'

        return term, sign, 'linear'

    def _parse_reprs_astA_astB_ip_C(self, test_form=None):
        r"""(AB, C)

        A and B are known.
        """

        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['A'], spk['B'], spk['C']

        if test_form == C:

            cpm = _VarPar_astA_astB_ip_tC(A, B, C)  # a root-array matrix

            v0 = C.ap().T
            term_ap = v0 @ cpm

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'
        return term, sign, 'linear'

    # ============== (AB, C) =====================================================================

    # -------- (AB, d(C)) --------------------------------------------------------------------------
    def _parse_reprs_AB_ip_dC(self, test_form):
        r""" Nonlinear, A, B and C are all unknown.

        Returns
        -------

        """
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['A'], spk['B'], spk['C']

        assert test_form in (A, B, C)

        multi_dimensional_array = _VarPar_AB_ip_dC(A, B, C)

        term = self._wft._factor * TermNonLinearOperatorAlgebraicProxy(
            multi_dimensional_array,
            [A, B, C]
        )
        sign = '+'

        term.set_test_form(test_form)

        return term, sign, 'nonlinear'

    def _parse_reprs_astA_B_ip_dC(self, test_form=None):
        r"""(AB, dC)

        A is known.
        """
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['A'], spk['B'], spk['C']

        if test_form == C:

            cpm = _VarPar_astA_B_ip_dtC(A, B, C)  # a root-array matrix

            v0 = C.ap().T
            v1 = B.ap()
            term_ap = v0 @ cpm @ v1

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'

        return term, sign, 'linear'

    def _parse_reprs_A_astB_ip_dC(self, test_form=None):
        r"""(AB, dC)

        B is known.
        """
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['A'], spk['B'], spk['C']

        if test_form == C:

            cpm = _VarPar_A_astB_ip_dtC(A, B, C)  # a root-array matrix

            v0 = C.ap().T
            v1 = A.ap()
            term_ap = v0 @ cpm @ v1

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'

        return term, sign, 'linear'

    def _parse_reprs_astA_astB_ip_dC(self, test_form=None):
        r"""(AB, dC)

        A and B are known.
        """

        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['A'], spk['B'], spk['C']

        if test_form == C:

            cpm = _VarPar_astA_astB_ip_dtC(A, B, C)  # a root-array matrix

            v0 = C.ap().T
            term_ap = v0 @ cpm

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'
        return term, sign, 'linear'

    # ============== (AB, dC) =====================================================================

    # -------- <AB|d(C)> --------------------------------------------------------------------------
    def _parse_reprs_AB_dp_dC(self, test_form):
        r""" Nonlinear, A, B and C are all unknown.

        Returns
        -------

        """
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['A'], spk['B'], spk['C']

        assert test_form in (A, B, C)

        multi_dimensional_array = _VarPar_AB_ip_dC(A, B, C)

        term = self._wft._factor * TermNonLinearOperatorAlgebraicProxy(
            multi_dimensional_array,
            [A, B, C]
        )
        sign = '+'

        term.set_test_form(test_form)

        return term, sign, 'nonlinear'

    def _parse_reprs_astA_B_dp_dC(self, test_form=None):
        r"""<AB|d(C)>

        A is known.
        """
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['A'], spk['B'], spk['C']

        if test_form == C:

            cpm = _VarPar_astA_B_dp_dtC(A, B, C)  # a root-array matrix

            v0 = C.ap().T
            v1 = B.ap()
            term_ap = v0 @ cpm @ v1

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'

        return term, sign, 'linear'

    def _parse_reprs_A_astB_dp_dC(self, test_form=None):
        r"""<AB|d(C)>

        B is known.
        """
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['A'], spk['B'], spk['C']

        if test_form == C:

            cpm = _VarPar_A_astB_dp_dtC(A, B, C)  # a root-array matrix

            v0 = C.ap().T
            v1 = A.ap()
            term_ap = v0 @ cpm @ v1

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'

        return term, sign, 'linear'

    def _parse_reprs_astA_astB_dp_dC(self, test_form=None):
        r"""<AB|d(C)>

        A and B are known.
        """

        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['A'], spk['B'], spk['C']

        if test_form == C:

            cpm = _VarPar_astA_astB_dp_dtC(A, B, C)  # a root-array matrix

            v0 = C.ap().T
            term_ap = v0 @ cpm

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'
        return term, sign, 'linear'

    # ============== <AB|d(C)> =====================================================================

    # (w x u, v) ---------------------------------------------------------------------------------
    def _parse_reprs_astA_x_astB_ip_C(self, test_form=None):
        """"""
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['a'], spk['b'], spk['c']

        if test_form == C:

            cpm = _VarPar_astA_x_astB_ip_tC(A, B, C)  # a root-array matrix

            v0 = C.ap().T
            term_ap = v0 @ cpm

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'

        return term, sign, 'linear'

    def _parse_reprs_astA_x_B_ip_C(self, test_form=None):
        """(A x B, C) where A is known! So this term is linear."""
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['a'], spk['b'], spk['c']

        if test_form == C:

            cpm = _VarPar_astA_x_B_ip_tC(A, B, C)  # a root-array matrix

            v0 = C.ap().T
            v1 = B.ap()
            term_ap = v0 @ cpm @ v1

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'

        return term, sign, 'linear'

    def _parse_reprs_A_x_astB_ip_C(self, test_form=None):
        """(A x B, C) where B is known! So this term is linear."""
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['a'], spk['b'], spk['c']

        if test_form == C:

            cpm = _VarPar_A_x_astB_ip_tC(A, B, C)  # a root-array matrix

            v0 = C.ap().T
            v1 = A.ap()
            term_ap = v0 @ cpm @ v1

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'

        return term, sign, 'linear'

    def _parse_reprs_A_x_B_ip_C(self, test_form):
        """(A x B, C),  this term is nonlinear."""
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['a'], spk['b'], spk['c']

        assert test_form in (A, B, C)

        multi_dimensional_array = _VarPar_A_x_B_ip_C(A, B, C)

        term = self._wft._factor * TermNonLinearOperatorAlgebraicProxy(
            multi_dimensional_array,
            [A, B, C]
        )
        sign = '+'

        term.set_test_form(test_form)

        return term, sign, 'nonlinear'

    # --- <A x B | C> ---------------------------------------------------------------------
    def _parse_reprs_astA_x_astB_dp_C(self, test_form):
        """"""
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['A'], spk['B'], spk['C']

        if test_form == C:

            cpm = _VarPar_astA_x_astB_dp_tC(A, B, C)  # a root-array matrix

            v0 = C.ap().T
            term_ap = v0 @ cpm

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'
        return term, sign, 'linear'

    def _parse_reprs_astA_x_B_dp_C(self, test_form):
        """"""
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['A'], spk['B'], spk['C']

        if test_form == C:

            cpm = _VarPar_astA_x_B_dp_tC(A, B, C)  # a root-array matrix

            v0 = C.ap().T
            v1 = B.ap()
            term_ap = v0 @ cpm @ v1

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'

        return term, sign, 'linear'

    def _parse_reprs_A_x_astB_dp_C(self, test_form):
        """"""
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['A'], spk['B'], spk['C']

        if test_form == C:

            cpm = _VarPar_A_x_astB_dp_tC(A, B, C)  # a root-array matrix

            v0 = C.ap().T
            v1 = A.ap()
            term_ap = v0 @ cpm @ v1

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'

        return term, sign, 'linear'

    def _parse_reprs_A_x_B_dp_C(self, test_form):
        r"""<A x B | C> where A, B, and C are all unknown. So, this is a nonlinear term."""
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['A'], spk['B'], spk['C']

        assert test_form in (A, B, C)

        multi_dimensional_array = _VarPar_A_x_B_dp_C(A, B, C)

        term = self._wft._factor * TermNonLinearOperatorAlgebraicProxy(
            multi_dimensional_array,
            [A, B, C]
        )
        sign = '+'

        term.set_test_form(test_form)

        return term, sign, 'nonlinear'

    # ---- <A x B | d(C)> ----------------------------------------------------------------
    def _parse_reprs_astA_x_astB_dp_dC(self, test_form):
        """<*A x *B | d(@C)>"""
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['A'], spk['B'], spk['C']
        dC = spk['dC']
        new_intermediate_root_form = dC.space.make_random_form()
        new_intermediate_root_form.degree = C.degree

        if test_form == C:

            cpm = _VarPar_astA_x_astB_dp_tC(A, B, new_intermediate_root_form)  # a root-array matrix
            dT_matrix = _VarPar_E(C, transpose=True)

            v0 = C.ap().T
            term_ap = v0 @ dT_matrix @ cpm

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'

        return term, sign, 'linear'

    # --- (A x B, C x D) -----------------------------------------------------------------
    def _parse_reprs_astA_x_astB__ip__astC_x_D(self, test_form):
        """(*A x *B, *C X D) where A, B, C are known."""
        spk = self._wft.___simple_pattern_keys___
        A, B, C, D = spk['A'], spk['B'], spk['C'], spk['D']

        if test_form == D:
            cpm = _VarPar_astA_x_astB__ip__astC_x_tD(A, B, C, D)

            v0 = D.ap().T
            term_ap = v0 @ cpm

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'

        return term, sign, 'linear'

    def _parse_reprs_A_x_astB__ip__astC_x_D(self, test_form):
        """(A x *B, *C X D) where  B, C are known."""
        spk = self._wft.___simple_pattern_keys___
        A, B, C, D = spk['A'], spk['B'], spk['C'], spk['D']

        if test_form == D:
            cpm = _VarPar_A_x_astB__ip__astC_x_tD(A, B, C, D)

            v0 = D.ap().T
            v1 = A.ap()
            term_ap = v0 @ cpm @ v1

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'

        return term, sign, 'linear'

    # -------- <A x B, C x D> -----------------------------------------------------------------
    def _parse_reprs_astA_x_astB__dp__astC_x_D(self, test_form):
        """<A x B | C X D> where A, B, C are known."""
        spk = self._wft.___simple_pattern_keys___
        A, B, C, D = spk['A'], spk['B'], spk['C'], spk['D']

        if test_form == D:
            cpm = _VarPar_astA_x_astB__dp__astC_x_tD(A, B, C, D)

            v0 = D.ap().T
            term_ap = v0 @ cpm

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'

        return term, sign, 'linear'

    # (A x B, d C) -------------------------------------------------------------------
    def _parse_reprs_astA_x_B_ip_dC(self, test_form):
        """(A x B, dC) where A is known! So this term is linear."""
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['a'], spk['b'], spk['c']
        dC = spk['dc']
        new_intermediate_root_form = dC.space.make_random_form()
        new_intermediate_root_form.degree = C.degree

        if test_form == C:

            cpm = _VarPar_astA_x_B_ip_tC(A, B, new_intermediate_root_form)  # a root-array matrix
            dT_matrix = _VarPar_E(C, transpose=True)

            v0 = C.ap().T
            v1 = B.ap()
            term_ap = v0 @ dT_matrix @ cpm @ v1

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'

        return term, sign, 'linear'

    def _parse_reprs_A_x_astB_ip_dC(self, test_form):
        """(A x B, dC) where B is known! So this term is linear."""
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['a'], spk['b'], spk['c']
        dC = spk['dc']
        new_intermediate_root_form = dC.space.make_random_form()
        new_intermediate_root_form.degree = C.degree

        if test_form == C:

            cpm = _VarPar_A_x_astB_ip_tC(A, B, new_intermediate_root_form)  # a root-array matrix
            dT_matrix = _VarPar_E(C, transpose=True)

            v0 = C.ap().T
            v1 = A.ap()
            term_ap = v0 @ dT_matrix @ cpm @ v1

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'

        return term, sign, 'linear'

    def _parse_reprs_astA_x_astB_ip_dC(self, test_form):
        """(A x B, dC), A and B are known. This will give a vector."""
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['a'], spk['b'], spk['c']
        dC = spk['dc']
        new_intermediate_root_form = dC.space.make_random_form()
        new_intermediate_root_form.degree = C.degree

        if test_form == C:

            cpm = _VarPar_astA_x_astB_ip_tC(A, B, new_intermediate_root_form)  # a root-array matrix
            dT_matrix = _VarPar_E(C, transpose=True)

            v0 = C.ap().T
            term_ap = v0 @ dT_matrix @ cpm

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'

        return term, sign, 'linear'

    def _parse_reprs_A_x_B_ip_dC(self, test_form):
        r""""""
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['a'], spk['b'], spk['c']
        dC = spk['dc']
        new_intermediate_root_form = dC.space.make_random_form()
        new_intermediate_root_form.degree = C.degree

        assert test_form in (A, B, C)

        multi_dimensional_array = _VarPar_AxB_ip_dC(A, B, C)

        term = self._wft._factor * TermNonLinearOperatorAlgebraicProxy(
            multi_dimensional_array,
            [A, B, C]
        )
        sign = '+'

        term.set_test_form(test_form)

        return term, sign, 'nonlinear'

    # (dA, B tp C) --------------------------------------------------------------------------
    def _parse_reprs_dastA_astA_tp_C(self, test_form):
        """"""
        spk = self._wft.___simple_pattern_keys___
        A, C = spk['A'], spk['C']

        if test_form == C:
            gA, tC = A, C

            cpm = _VarPar_dastA_astA_tp_tC(gA, tC)  # a root-array matrix

            v0 = tC.ap().T
            term_ap = v0 @ cpm

        else:
            raise Exception

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'

        return term, sign, 'linear'

    def _parse_reprs_dastA_tB_tp_astA(self, test_form):
        """"""
        spk = self._wft.___simple_pattern_keys___
        A, B = spk['A'], spk['B']

        if test_form == B:
            gA, tB = A, B

            cpm = _VarPar_dastA_tB_tp_astA(gA, tB)  # a root-array matrix

            v0 = tB.ap().T
            term_ap = v0 @ cpm

        else:
            raise Exception

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'

        return term, sign, 'linear'

    def _parse_reprs_dA_astB_tp_astB(self, test_form):
        """"""
        spk = self._wft.___simple_pattern_keys___
        A, B = spk['A'], spk['B']
        if test_form == A:
            tA, gB = A, B

            cpm = _VarPar_dtA_astB_tp_astB(tA, gB)  # a root-array matrix

            v0 = tA.ap().T
            term_ap = v0 @ cpm

        else:
            raise Exception

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'

        return term, sign, 'linear'

    def _parse_reprs_dA_B_tp_C__1Known(self, test_form):
        """"""
        spk = self._wft.___simple_pattern_keys___
        A, B, C, kf = spk['A'], spk['B'], spk['C'], spk['K']
        tf = test_form
        unknown = None
        for _ in (A, B, C):
            if kf is _ or tf is _:
                pass
            else:
                assert unknown is None, f'must have found only one unknown form.'
                unknown = _
        assert unknown is not None, f'must have found the unknown form.'
        cpm = _VarPar_dA_B_tp_C__1Known(A, B, C, kf, tf)
        v0 = tf.ap().T
        v1 = unknown.ap()
        term_ap = v0 @ cpm @ v1
        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'
        return term, sign, 'linear'

    def _parse_reprs_dA_B_tp_C__2Known(self, test_form):
        spk = self._wft.___simple_pattern_keys___
        A, B, C, kf1, kf2 = spk['A'], spk['B'], spk['C'], spk['K1'], spk['K2']
        cpm = _VarPar_dA_B_tp_C__2Known(A, B, C, kf1, kf2, test_form)
        v0 = test_form.ap().T
        term_ap = v0 @ cpm
        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'
        return term, sign, 'linear'

    def _parse_reprs_dA_B_tp_C(self, test_form):
        """(dA, B otimes C), nonlinear"""
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['A'], spk['B'], spk['C']
        assert A is not B and B is not C and A is not C, f"A, B, C must be different."

        assert test_form in (A, B, C)

        multi_dimensional_array = _VarPar_dA_B_tp_C(A, B, C)

        term = self._wft._factor * TermNonLinearOperatorAlgebraicProxy(
            multi_dimensional_array,
            [A, B, C]
        )
        sign = '+'

        term.set_test_form(test_form)

        return term, sign, 'nonlinear'

    # --- (A, B otimes C) --------------------------------------------------------------
    def _parse_reprs_A_B_tp_C__1Known(self, test_form):
        """"""
        spk = self._wft.___simple_pattern_keys___
        A, B, C, kf = spk['A'], spk['B'], spk['C'], spk['K']
        tf = test_form
        unknown = None
        for _ in (A, B, C):
            if kf is _ or tf is _:
                pass
            else:
                assert unknown is None, f'must have found only one unknown form.'
                unknown = _
        assert unknown is not None, f'must have found the unknown form.'
        cpm = _VarPar_A_B_tp_C__1Known(A, B, C, kf, tf)
        v0 = tf.ap().T
        v1 = unknown.ap()
        term_ap = v0 @ cpm @ v1
        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'
        return term, sign, 'linear'

    def _parse_reprs_A_B_tp_C__2Known(self, test_form):
        spk = self._wft.___simple_pattern_keys___
        A, B, C, kf1, kf2 = spk['A'], spk['B'], spk['C'], spk['K1'], spk['K2']
        cpm = _VarPar_A_B_tp_C__2Known(A, B, C, kf1, kf2, test_form)
        v0 = test_form.ap().T
        term_ap = v0 @ cpm
        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'
        return term, sign, 'linear'

    def _parse_reprs_A_B_tp_C(self, test_form):
        """(dA, B otimes C), nonlinear"""
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['A'], spk['B'], spk['C']
        assert A is not B and B is not C and A is not C, f"A, B, C must be different."

        assert test_form in (A, B, C)

        multi_dimensional_array = _VarPar_A_B_tp_C(A, B, C)

        term = self._wft._factor * TermNonLinearOperatorAlgebraicProxy(
            multi_dimensional_array,
            [A, B, C]
        )
        sign = '+'

        term.set_test_form(test_form)

        return term, sign, 'nonlinear'

    # (bundle form, special diagonal bundle form)------------------------------------------------------

    def _parse_reprs_dbdb(self, test_form):
        """"""
        spk = self._wft.___simple_pattern_keys___
        db0, bf1 = spk['db0'], spk['bf1']

        dbf1 = bf1.exterior_derivative()
        dbf1._degree = bf1._degree

        if test_form == bf1:
            mass_matrix = _VarPar_l2_inner_product_db_bf(db0, dbf1, transpose=True)
            dT_matrix = _VarPar_E(bf1, transpose=True)
            v0 = db0.ap()
            v1 = bf1.ap().T
            term_ap = v1 @ dT_matrix @ mass_matrix @ v0

        else:
            mass_matrix = _VarPar_l2_inner_product_db_bf(db0, dbf1, transpose=False)
            d_matrix = _VarPar_E(bf1)
            v0 = db0.ap().T
            v1 = bf1.ap()
            term_ap = v0 @ mass_matrix @ d_matrix @ v1

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'
        return term, sign, 'linear'

    # ------ (A, d(pi(B)) ----------------------------------------------------------------------------
    def _parse_reprs___d_pi(self, test_form):
        """ <A , d pi B>, d pi B is in the same space and is of same degree of A.

        A, B are root-forms.
        """
        spk = self._wft.___simple_pattern_keys___
        A, B = spk['A'], spk['B']

        mass_matrix = _VarPar_M(A.space, A.space, A.degree, A.degree)

        from src.spaces.operators import codifferential
        pi_B_space = codifferential(A.space)
        pi_B_degree = A.degree

        if test_form == B:
            dT = _VarPar_E((pi_B_space, pi_B_degree), transpose=True)
            pT = _VarPar_P(
                (B.space, pi_B_space),
                (B.degree, pi_B_degree),
                transpose=True
            )
            v0 = B.ap().T
            v1 = A.ap()
            term_ap = v0 @ pT @ dT @ mass_matrix @ v1

        else:
            assert test_form is A, f'must be!'
            d = _VarPar_E((pi_B_space, pi_B_degree), transpose=False)
            p = _VarPar_P(
                (B.space, pi_B_space),
                (B.degree, pi_B_degree),
                transpose=False
            )
            v0 = A.ap().T
            v1 = B.ap()
            term_ap = v0 @ mass_matrix @ d @ p @ v1

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'
        return term, sign, 'linear'

    def _parse_reprs_d_star_A_ip_B(self, test_form):
        """(d(*A), B)"""
        spk = self._wft.___simple_pattern_keys___
        A, B = spk['A'], spk['B']

        M = _VarPar_M(B.space, B.space, B.degree, B.degree)

        from src.spaces.operators import codifferential
        star_A_space = codifferential(B.space)
        star_A_degree = A.degree  # This Hodge could be inaccurate.

        if test_form == B:
            E = _VarPar_E((star_A_space, star_A_degree), transpose=False)
            H = _VarPar_H(
                A.space, A.degree,
                star_A_space, star_A_degree
            )
            v0 = B.ap().T
            v1 = A.ap()
            term_ap = v0 @ M @ E @ H @ v1
        else:
            raise NotImplementedError()

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'
        return term, sign, 'linear'
