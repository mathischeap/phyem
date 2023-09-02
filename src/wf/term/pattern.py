# -*- coding: utf-8 -*-
r"""
"""
from src.form.operators import _parse_related_time_derivative
from src.form.main import _global_root_forms_lin_dict
from src.form.operators import time_derivative, d
from src.config import _global_operator_lin_repr_setting
from src.form.others import _find_form, _find_root_forms_through_lin_repr
from src.config import _non_root_lin_sep
from src.config import _wf_term_default_simple_patterns as _simple_patterns
from src.form.parameters import ConstantScalar0Form


def _inner_simpler_pattern_examiner_scalar_valued_forms(factor, f0, f1, extra_info):
    """"""
    if factor.__class__ is ConstantScalar0Form:

        # (codifferential sf, sf) ----------------------------------------------------------------------
        lin_codifferential = _global_operator_lin_repr_setting['codifferential']
        if f0._lin_repr[:len(lin_codifferential)] == lin_codifferential:
            return _simple_patterns['(cd,)'], None

        # (partial_time_derivative of root-sf, sf) -----------------------------------------------------
        lin_td = _global_operator_lin_repr_setting['time_derivative']
        if f0._lin_repr[:len(lin_td)] == lin_td:
            bf0 = _find_form(f0._lin_repr, upon=time_derivative)
            if bf0.is_root and _parse_related_time_derivative(f1) == list():
                return _simple_patterns['(pt,)'], {
                    'rsf0': bf0,   # root-scalar-form-0
                    'rsf1': f1,    # root-scalar-form-1
                }

        # (root-sf, root-sf) ---------------------------------------------------------------------------
        if f0.is_root() and f1.is_root():
            return _simple_patterns['(rt,rt)'], {
                    'rsf0': f0,   # root-scalar-form-0
                    'rsf1': f1,   # root-scalar-form-1
                }
        else:
            pass

        # (d of root-sf, root-sf) ---------------------------------------------------------------------
        lin_d = _global_operator_lin_repr_setting['d']
        if f0._lin_repr[:len(lin_d)] == lin_d:
            bf0 = _find_form(f0._lin_repr, upon=d)
            if bf0 is None:
                pass
            elif bf0.is_root() and f1.is_root():
                return _simple_patterns['(d,)'], {
                    'rsf0': bf0,   # root-scalar-form-0
                    'rsf1': f1,    # root-scalar-form-1
                }
            else:
                pass

        # (root-sf, d of root-sf) -------------------------------------------------------------------
        lin_d = _global_operator_lin_repr_setting['d']
        if f1._lin_repr[:len(lin_d)] == lin_d:
            bf1 = _find_form(f1._lin_repr, upon=d)
            if bf1 is None:
                pass
            elif f0.is_root() and bf1.is_root():
                return _simple_patterns['(,d)'], {
                    'rsf0': f0,     # root-scalar-form-0
                    'rsf1': bf1,    # root-scalar-form-1
                }
            else:
                pass

        # <A, d(pi(B))> --------------------------------------------------------------------------------
        projection_lin = _global_operator_lin_repr_setting['projection']
        if projection_lin in f1._lin_repr and lin_d in f1._lin_repr:
            root_forms_in_f1 = _find_root_forms_through_lin_repr(f1._lin_repr)
            # Both A and B are root-form
            if len(root_forms_in_f1) == 1 and f0._is_root:
                pi_B = _find_form(f1._lin_repr, upon=d)
                return _simple_patterns['(,d-pi)'], {
                    'A': f0,                     # root-form A
                    'B': root_forms_in_f1[0],    # root-form B
                    'pi(B)': pi_B
                }

        # (a x b, c) types term, where x is cross product, and a, b, c are root-scalar-valued forms ----
        cross_product_lin = _global_operator_lin_repr_setting['cross_product']

        if cross_product_lin in f0._lin_repr and f0._lin_repr.count(cross_product_lin) == 1:
            # it is like `a` x `b` for f0
            a_lin, b_lin = f0._lin_repr.split(cross_product_lin)

            f_a = _find_form(a_lin)
            f_b = _find_form(b_lin)

            if (f_a.is_root() and f_b.is_root() and f1.is_root() and
                    (f_a is not f_b) and (f_a is not f1) and (f_b is not f1)):
                # a, b and c are all root-forms. This is good!

                if 'known-cross-product-form' in extra_info:
                    known_forms = extra_info['known-cross-product-form']
                    if known_forms is f_a:
                        return _simple_patterns['(*x,)'], {
                            'a': f_a,
                            'b': f_b,
                            'c': f1
                        }

                    elif known_forms is f_b:

                        return _simple_patterns['(x*,)'], {
                            'a': f_a,
                            'b': f_b,
                            'c': f1
                        }

                    elif isinstance(known_forms, (list, tuple)) and len(known_forms) == 2:

                        kf0, kf1 = known_forms

                        if kf0 is f_a and kf1 is f_b:

                            return _simple_patterns['(*x*,)'], {
                                'a': f_a,
                                'b': f_b,
                                'c': f1
                            }

                    else:
                        raise Exception

                else:
                    # this term will be a nonlinear one! Take care it in the future!

                    return _simple_patterns['(x,)'], {
                        'a': f_a,
                        'b': f_b,
                        'c': f1
                    }

            else:
                pass

        return '', None

    else:
        raise NotImplementedError(f'Not implemented for factor={factor}')


def _inner_simpler_pattern_examiner_bundle_valued_forms(factor, f0, f1, extra_info):
    """"""
    if factor.__class__ is ConstantScalar0Form:

        # (codifferential sf, sf) ---------------------------------------------------------------------
        lin_codifferential = _global_operator_lin_repr_setting['codifferential']
        if f0._lin_repr[:len(lin_codifferential)] == lin_codifferential:
            return _simple_patterns['(cd,)'], None

        # (partial_time_derivative of root-sf, sf) ----------------------------------------------------
        lin_td = _global_operator_lin_repr_setting['time_derivative']
        if f0._lin_repr[:len(lin_td)] == lin_td:
            bf0 = _find_form(f0._lin_repr, upon=time_derivative)
            if bf0.is_root and _parse_related_time_derivative(f1) == list():
                return _simple_patterns['(pt,)'], {
                    'rsf0': bf0,   # root-scalar-form-0
                    'rsf1': f1,    # root-scalar-form-1
                }

        # (root-sf, root-sf) ---------------------------------------------------------------------------
        if f0.is_root() and f1.is_root():
            return _simple_patterns['(rt,rt)'], {
                'rsf0': f0,   # root-scalar-form-0
                'rsf1': f1,   # root-scalar-form-1
            }

        # (d of root-sf, root-sf) ----------------------------------------------------------------------
        lin_d = _global_operator_lin_repr_setting['d']
        if f0._lin_repr[:len(lin_d)] == lin_d:
            bf0 = _find_form(f0._lin_repr, upon=d)
            if bf0 is None:
                pass
            elif bf0.is_root() and f1.is_root():
                return _simple_patterns['(d,)'], {
                    'rsf0': bf0,   # root-scalar-form-0
                    'rsf1': f1,    # root-scalar-form-1
                }
            else:
                pass

        # (root-sf, d of root-sf) --------------------------------------------------------------------
        lin_d = _global_operator_lin_repr_setting['d']
        if f1._lin_repr[:len(lin_d)] == lin_d:
            bf1 = _find_form(f1._lin_repr, upon=d)
            if bf1 is None:
                pass
            elif f0.is_root() and bf1.is_root():
                return _simple_patterns['(,d)'], {
                    'rsf0': f0,     # root-scalar-form-0
                    'rsf1': bf1,    # root-scalar-form-1
                }
            else:
                pass

        tensor_product_lin = _global_operator_lin_repr_setting['tensor_product']

        # (d A, B tp C) --------------------------------------------------------------------------
        if (f0._lin_repr[:len(lin_d)] == lin_d and
                tensor_product_lin in f1._lin_repr and f1._lin_repr.count(tensor_product_lin) == 1):

            A = _find_form(f0._lin_repr, upon=d)
            B_lin_repr, C_lin_repr = f1._lin_repr.split(tensor_product_lin)
            B = _find_form(B_lin_repr)
            C = _find_form(C_lin_repr)

            # A, B, C are all root forms.
            if A.is_root() and B.is_root() and C.is_root():
                # (d A, A tp C).
                if (A is B) and (A is not C):
                    if 'known-tensor-product-form' in extra_info:
                        known_forms = extra_info['known-tensor-product-form']
                        if known_forms is A:
                            return _simple_patterns['(d0*,0*tp)'], {
                                'A': A,
                                'C': C
                            }

                # (d A, B tp A).
                elif (A is C) and (A is not B):
                    if 'known-tensor-product-form' in extra_info:
                        known_forms = extra_info['known-tensor-product-form']
                        if known_forms is A:
                            return _simple_patterns['(d0*,tp0*)'], {
                                'A': A,
                                'B': B
                            }

                # (d A, B tp B).
                elif (B is C) and (A is not B):
                    if 'known-tensor-product-form' in extra_info:
                        known_forms = extra_info['known-tensor-product-form']
                        if known_forms is B:
                            return _simple_patterns['(d,0*tp0*)'], {
                                'A': A,
                                'B': B
                            }

                elif (A is not C) and (A is not B) and (B is not C):
                    if 'known-tensor-product-form' in extra_info:
                        known_forms = extra_info['known-tensor-product-form']
                        if known_forms in (A, B, C):  # only one known
                            return _simple_patterns['(d,tp):1K'], {
                                'A': A,
                                'B': B,
                                'C': C,
                                'K': known_forms,
                            }
                        elif len(known_forms) == 2:  # two known forms
                            return _simple_patterns['(d,tp):2K'], {
                                'A': A,
                                'B': B,
                                'C': C,
                                'K1': known_forms[0],
                                'K2': known_forms[1],
                            }

                    else:  # nonlinear
                        return _simple_patterns['(d,tp)'], {
                            'A': A,
                            'B': B,
                            'C': C,
                        }

        # (A, B tp C) --------------------------------------------------------------------------
        if tensor_product_lin in f1._lin_repr and f1._lin_repr.count(tensor_product_lin) == 1:
            A = _find_form(f0._lin_repr)
            B_lin_repr, C_lin_repr = f1._lin_repr.split(tensor_product_lin)
            B = _find_form(B_lin_repr)
            C = _find_form(C_lin_repr)

            # A, B, C are all root forms.
            if A.is_root() and B.is_root() and C.is_root():
                # A, B, C are all different root forms.
                if (A is not C) and (A is not B) and (B is not C):
                    if 'known-tensor-product-form' in extra_info:
                        known_forms = extra_info['known-tensor-product-form']
                        if known_forms in (A, B, C):  # only one known
                            return _simple_patterns['(,tp):1K'], {
                                'A': A,
                                'B': B,
                                'C': C,
                                'K': known_forms,
                            }
                        elif len(known_forms) == 2:  # two known forms
                            return _simple_patterns['(,tp):2K'], {
                                'A': A,
                                'B': B,
                                'C': C,
                                'K1': known_forms[0],
                                'K2': known_forms[1],
                            }

                    else:  # nonlinear
                        return _simple_patterns['(,tp)'], {
                            'A': A,
                            'B': B,
                            'C': C,
                        }

        # ========= No pattern found !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! =======================
        return '', None

    else:
        raise NotImplementedError(f'Not implemented for factor={factor}, extra_info={extra_info}.')


def _inner_simpler_pattern_examiner_diagonal_bundle_valued_forms(factor, f0, f1, extra_info):
    """"""
    if factor.__class__ is ConstantScalar0Form:

        # (root-sf, d of root-sf) -------------------------------------------
        lin_d = _global_operator_lin_repr_setting['d']
        if f1._lin_repr[:len(lin_d)] == lin_d:
            bf1 = _find_form(f1._lin_repr, upon=d)
            if bf1 is None:
                pass
            elif f0.is_root() and bf1.is_root():
                return _simple_patterns['(<db>,d<b>)'], {
                    'db0': f0,
                    'bf1': bf1,
                }
            else:
                pass

        return '', None

    else:
        raise NotImplementedError(f'Not implemented for factor={factor}, extra_info={extra_info}.')


def _dp_simpler_pattern_examiner_scalar_valued_forms(factor, f0, f1):
    """ """
    if factor.__class__.__name__ == 'ConstantScalar0Form':
        lin_tr = _global_operator_lin_repr_setting['trace']
        lin_hodge = _global_operator_lin_repr_setting['Hodge']

        lin = lin_tr + _non_root_lin_sep[0] + lin_hodge
        if f0._lin_repr[:len(lin)] == lin and \
                f0._lin_repr[-len(_non_root_lin_sep[1]):] == _non_root_lin_sep[1] and \
                f1._lin_repr[:len(lin_tr)] == lin_tr:

            bf0_lr = f0._lin_repr[len(lin):-len(_non_root_lin_sep[1])]
            bf1_lr = f1._lin_repr[len(lin_tr):]

            if bf0_lr in _global_root_forms_lin_dict and bf1_lr in _global_root_forms_lin_dict:
                bf0 = _global_root_forms_lin_dict[bf0_lr]
                bf1 = _global_root_forms_lin_dict[bf1_lr]

                return _simple_patterns['<tr star | tr >'], {
                    'rsf0': bf0,   # root-scalar-form-0
                    'rsf1': bf1,   # root-scalar-form-1
                }

        return '', None
    else:
        raise NotImplementedError(f'Not implemented for factor={factor}')
