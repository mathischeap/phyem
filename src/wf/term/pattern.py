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
    """ """
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

            if f_a.is_root() and f_b.is_root() and f1.is_root():
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
                        pass

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
            else:
                pass

        else:
            pass

        return '', None
    else:
        raise NotImplementedError(f'Not implemented for factor={factor}')
