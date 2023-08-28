# -*- coding: utf-8 -*-
r"""
"""
from src.form.operators import _parse_related_time_derivative
from src.form.main import _global_root_forms_lin_dict
from src.form.operators import time_derivative, d
from src.config import _global_operator_lin_repr_setting
from src.form.others import _find_form
from src.config import _non_root_lin_sep
from src.config import _wf_term_default_simple_patterns as _simple_patterns
from src.form.parameters import ConstantScalar0Form


def _inner_simpler_pattern_examiner_scalar_valued_forms(factor, f0, f1, extra_info):
    """"""
    if factor.__class__ is ConstantScalar0Form:

        # (codifferential sf, sf) -------------------------------------------
        lin_codifferential = _global_operator_lin_repr_setting['codifferential']
        if f0._lin_repr[:len(lin_codifferential)] == lin_codifferential:
            return _simple_patterns['(cd,)'], None

        # (partial_time_derivative of root-sf, sf) --------------------------
        lin_td = _global_operator_lin_repr_setting['time_derivative']
        if f0._lin_repr[:len(lin_td)] == lin_td:
            bf0 = _find_form(f0._lin_repr, upon=time_derivative)
            if bf0.is_root and _parse_related_time_derivative(f1) == list():
                return _simple_patterns['(pt,)'], {
                    'rsf0': bf0,   # root-scalar-form-0
                    'rsf1': f1,    # root-scalar-form-1
                }

        # (root-sf, root-sf) ------------------------------------------------
        if f0.is_root() and f1.is_root():
            return _simple_patterns['(rt,rt)'], {
                    'rsf0': f0,   # root-scalar-form-0
                    'rsf1': f1,   # root-scalar-form-1
                }
        else:
            pass

        # (d of root-sf, root-sf) -------------------------------------------
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

        # (root-sf, d of root-sf) -------------------------------------------
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

        # (codifferential sf, sf) -------------------------------------------
        lin_codifferential = _global_operator_lin_repr_setting['codifferential']
        if f0._lin_repr[:len(lin_codifferential)] == lin_codifferential:
            return _simple_patterns['(cd,)'], None

        # (partial_time_derivative of root-sf, sf) --------------------------
        lin_td = _global_operator_lin_repr_setting['time_derivative']
        if f0._lin_repr[:len(lin_td)] == lin_td:
            bf0 = _find_form(f0._lin_repr, upon=time_derivative)
            if bf0.is_root and _parse_related_time_derivative(f1) == list():
                return _simple_patterns['(pt,)'], {
                    'rsf0': bf0,   # root-scalar-form-0
                    'rsf1': f1,    # root-scalar-form-1
                }

        # (root-sf, root-sf) ------------------------------------------------
        if f0.is_root() and f1.is_root():
            return _simple_patterns['(rt,rt)'], {
                'rsf0': f0,   # root-scalar-form-0
                'rsf1': f1,   # root-scalar-form-1
            }

        # (d of root-sf, root-sf) -------------------------------------------
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

        # (root-sf, d of root-sf) -------------------------------------------
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

        # (a tp b, _)
        if tensor_product_lin in f0._lin_repr and f0._lin_repr.count(tensor_product_lin) == 1 and f1.is_root():

            # it is like `a` tp `b` for f0
            a_lin, b_lin = f0._lin_repr.split(tensor_product_lin)

            f_a = _find_form(a_lin)
            f_b = _find_form(b_lin)

            # (`a` tp `b`, f1), a, b, f1 are all root.
            if f_a.is_root() and f_b.is_root():
                if f_a is f_b:  # (`a` tp `a`, f1)
                    if 'known-tensor-product-form' in extra_info:
                        known_forms = extra_info['known-tensor-product-form']
                        if known_forms is f_a:
                            return _simple_patterns['(0*tp0*,)'], {
                                'a': f_a,
                                'c': f1
                            }

                    else:
                        # this term will be a nonlinear one
                        return _simple_patterns['(0tp0,)'], {
                            'a': f_a,
                            'c': f1
                        }
                else:
                    if 'known-tensor-product-form' in extra_info:
                        known_forms = extra_info['known-tensor-product-form']

                        if known_forms is f_a:
                            return _simple_patterns['(*tp,)'], {
                                'a': f_a,
                                'b': f_b,
                                'c': f1
                            }

                    else:
                        pass

            else:
                pass

        # (d bf0, b tp c)
        if (f0._lin_repr[:len(lin_d)] == lin_d and
                tensor_product_lin in f1._lin_repr and f1._lin_repr.count(tensor_product_lin) == 1):

            a_lin, b_lin = f1._lin_repr.split(tensor_product_lin)

            f_b = _find_form(a_lin)
            f_c = _find_form(b_lin)

            bf0 = _find_form(f0._lin_repr, upon=d)

            if bf0.is_root() and f_b.is_root() and f_c.is_root():
                # bf0 b c are different.
                if (bf0 is not f_b) and (bf0 is not f_c) and (f_b is not f_c):
                    if 'known-tensor-product-form' in extra_info:
                        known_forms = extra_info['known-tensor-product-form']

                        if known_forms is f_b:
                            return _simple_patterns['(d,*tp)'], {
                                'a': bf0,
                                'b': f_b,
                                'c': f_c
                            }

                        elif known_forms is bf0:
                            return _simple_patterns['(d*,tp)'], {
                                'a': bf0,
                                'b': f_b,
                                'c': f_c
                            }
                        else:
                            pass

                    else:
                        pass

                # (d bf0, bf0 tp c) are different.
                elif (bf0 is f_b) and (bf0 is not f_c):
                    if 'known-tensor-product-form' in extra_info:
                        known_forms = extra_info['known-tensor-product-form']
                        if known_forms is bf0:
                            return _simple_patterns['(d0*,0*tp)'], {
                                'a': bf0,
                                'c': f_c
                            }
                    else:
                        # this term will be a nonlinear one
                        return _simple_patterns['(d0,0tp)'], {
                            'a': bf0,
                            'c': f_c
                        }

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
