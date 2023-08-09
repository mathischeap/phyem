# -*- coding: utf-8 -*-
"""
"""
from src.config import _wf_term_default_simple_patterns as _simple_patterns
from src.spaces.ap import _parse_l2_inner_product_mass_matrix
from src.spaces.ap import _parse_d_matrix
from src.spaces.ap import _parse_boundary_dp_vector
from src.spaces.ap import _parse_astA_x_B_ip_tC
from src.spaces.ap import _parse_A_x_astB_ip_tC
from src.form.parameters import ConstantScalar0Form
from tools.frozen import Frozen


class TermLinearAlgebraicProxy(Frozen):
    """It is basically a wrapper of a (1, 1) abstract array."""

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
            elif sp == _simple_patterns['(d,)']:
                return self._parse_reprs_d_(test_form=test_form)
            elif sp == _simple_patterns['(,d)']:
                return self._parse_reprs__d(test_form=test_form)
            elif sp == _simple_patterns['<tr star | tr >']:
                return self._parse_reprs_tr_star_star(test_form=test_form)
            elif sp == _simple_patterns['(*x,)']:
                return self._parse_reprs_astA_x_B_ip_C(test_form=test_form)
            elif sp == _simple_patterns['(x*,)']:
                return self._parse_reprs_A_x_astB_ip_C(test_form=test_form)
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
        mass_matrix = _parse_l2_inner_product_mass_matrix(s0, s1, d0, d1)
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
        return term, sign

    def _parse_reprs_rt_rt(self, test_form=None):
        """"""
        f0, f1 = self._wft._f0, self._wft._f1
        s0 = f0.space
        s1 = f1.space
        d0 = f0._degree
        d1 = f1._degree
        mass_matrix = _parse_l2_inner_product_mass_matrix(s0, s1, d0, d1)
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
        return term, sign

    def _parse_reprs_d_(self, test_form=None):
        """"""
        spk = self._wft.___simple_pattern_keys___
        bf0 = spk['rsf0']
        d0 = bf0._degree
        s0 = self._wft._f0.space
        s1 = self._wft._f1.space
        d1 = self._wft._f1._degree
        mass_matrix = _parse_l2_inner_product_mass_matrix(s0, s1, d0, d1)

        if test_form == self._wft._f1:
            d_matrix = _parse_d_matrix(bf0)
            v0 = bf0.ap()
            v1 = self._wft._f1.ap().T
            term_ap = v1 @ mass_matrix @ d_matrix @ v0
        else:
            dT_matrix = _parse_d_matrix(bf0, transpose=True)
            v0 = bf0.ap().T
            v1 = self._wft._f1.ap()
            term_ap = v0 @ dT_matrix @ mass_matrix @ v1
        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'
        return term, sign

    def _parse_reprs__d(self, test_form=None):
        """"""
        s0 = self._wft._f0.space
        d0 = self._wft._f0._degree

        spk = self._wft.___simple_pattern_keys___
        bf1 = spk['rsf1']
        d1 = bf1._degree
        s1 = self._wft._f1.space
        mass_matrix = _parse_l2_inner_product_mass_matrix(s0, s1, d0, d1)

        if test_form == bf1:
            dT_matrix = _parse_d_matrix(bf1, transpose=True)
            v0 = self._wft._f0.ap()
            v1 = bf1.ap().T
            term_ap = v1 @ dT_matrix @ mass_matrix @ v0
        else:
            d_matrix = _parse_d_matrix(bf1)
            v0 = self._wft._f0.ap().T
            v1 = bf1.ap()
            term_ap = v0 @ mass_matrix @ d_matrix @ v1
        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'
        return term, sign

    def _parse_reprs_tr_star_star(self, test_form=None):
        """<tr star bf0, tr bf1>"""
        spk = self._wft.___simple_pattern_keys___
        bf0 = spk['rsf0']
        bf1 = spk['rsf1']

        v1 = bf1.ap()
        # s1 = bf1.space
        boundary_wedge_vector = _parse_boundary_dp_vector(bf0, bf1)

        if test_form == bf1:  # when bf1 is the testing form.
            term_ap = v1.T @ boundary_wedge_vector
        else:
            raise NotImplementedError()

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'
        return term, sign

    def _parse_reprs_astA_x_B_ip_C(self, test_form=None):
        """(A x B, C) where A is known! So this term is linear."""
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['a'], spk['b'], spk['c']

        if test_form == C:

            cpm = _parse_astA_x_B_ip_tC(A, B, C)  # a root-array matrix

            v0 = C.ap().T
            v1 = B.ap()
            term_ap = v0 @ cpm @ v1

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'

        return term, sign

    def _parse_reprs_A_x_astB_ip_C(self, test_form=None):
        """(A x B, C) where B is known! So this term is linear."""
        spk = self._wft.___simple_pattern_keys___
        A, B, C = spk['a'], spk['b'], spk['c']

        if test_form == C:

            cpm = _parse_A_x_astB_ip_tC(A, B, C)  # a root-array matrix

            v0 = C.ap().T
            v1 = A.ap()
            term_ap = v0 @ cpm @ v1

        else:
            raise Exception('TO BE IMPLEMENTED!')  # better not to use NotImplementedError

        term = self._wft._factor * TermLinearAlgebraicProxy(term_ap)
        sign = '+'

        return term, sign
