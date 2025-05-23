# -*- coding: utf-8 -*-
r"""
"""

from src.spaces.operators import wedge as space_wedge
from src.spaces.operators import Hodge as space_Hodge
from src.spaces.operators import d as space_d
from src.spaces.operators import codifferential as space_codifferential
from src.spaces.operators import cross_product as space_cross_product
from src.spaces.operators import Cross_Product as space_Cross_Product
from src.spaces.operators import convect as space_convect
from src.spaces.operators import tensor_product as space_tensor_product
from src.spaces.operators import project_to as space_project_to

from src.config import _global_operator_lin_repr_setting
from src.config import _global_operator_sym_repr_setting

from src.config import _non_root_lin_sep

_time_derivative_related_operators = {
    'time_derivative': _global_operator_lin_repr_setting['time_derivative'],
}


def _parse_related_time_derivative(f):
    """"""
    related = list()
    for op_name in _time_derivative_related_operators:
        op_lin_rp = _time_derivative_related_operators[op_name]
        if op_lin_rp in f._lin_repr:
            related.append(op_name)
    return related


def _project_to(from_f, to_space):  # can be called only from a form. If used it somewhere else, wrap it.
    """"""
    from_space = from_f.space
    to_space = space_project_to(from_space, to_space)

    lr = from_f._lin_repr
    sr = from_f._sym_repr

    op_lin_repr = _global_operator_lin_repr_setting['projection']
    sr_operator = _global_operator_sym_repr_setting['projection']

    if from_f.is_root():
        lr = op_lin_repr + lr
    else:
        lr = op_lin_repr + _non_root_lin_sep[0] + lr + _non_root_lin_sep[1]

    if from_f.is_root():
        sr = sr_operator + sr
    else:
        sr = sr_operator + r"\left(" + sr + r"\right)"

    f = from_f.__class__(
        to_space,  # space
        sr,  # symbolic representation
        lr,
        False,
    )

    return f


def wedge(f1, f2):
    """f1 wedge f2"""
    s1 = f1.space
    s2 = f2.space

    wedge_space = space_wedge(s1, s2)  # if this is not possible, return NotImplementedError

    lr_term1 = f1._lin_repr
    lr_term2 = f2._lin_repr
    lr_operator = _global_operator_lin_repr_setting['wedge']

    sr_term1 = f1._sym_repr
    sr_term2 = f2._sym_repr
    sr_operator = _global_operator_sym_repr_setting['wedge']

    if f1.is_root():
        pass
    else:
        lr_term1 = _non_root_lin_sep[0] + lr_term1 + _non_root_lin_sep[1]
        sr_term1 = r'\left(' + sr_term1 + r'\right)'
    if f2.is_root():
        pass
    else:
        lr_term2 = _non_root_lin_sep[0] + lr_term2 + _non_root_lin_sep[1]
        sr_term2 = r'\left(' + sr_term2 + r'\right)'
    lin_repr = lr_term1 + lr_operator + lr_term2
    sym_repr = sr_term1 + sr_operator + sr_term2

    f = f1.__class__(
        wedge_space,  # space
        sym_repr,  # symbolic representation
        lin_repr,
        False
    )

    return f


def Hodge(f):
    """Metric Hodge of a form."""
    hs = space_Hodge(f.space)

    lr = f._lin_repr
    sr = f._sym_repr

    op_lin_repr = _global_operator_lin_repr_setting['Hodge']
    sr_operator = _global_operator_sym_repr_setting['Hodge']

    if f.is_root():
        lr = op_lin_repr + lr
    else:
        lr = op_lin_repr + _non_root_lin_sep[0] + lr + _non_root_lin_sep[1]

    if f.is_root():
        sr = sr_operator + sr
    else:
        sr = sr_operator + r"\left(" + sr + r"\right)"

    f = f.__class__(
        hs,  # space
        sr,  # symbolic representation
        lr,
        False,
    )

    return f


def d(f):
    """Metric Hodge of a form."""
    ds = space_d(f.space)

    lr = f._lin_repr
    sr = f._sym_repr

    op_lin_repr = _global_operator_lin_repr_setting['d']
    sr_operator = _global_operator_sym_repr_setting['d']

    if f.is_root():
        lr = op_lin_repr + lr
    else:
        lr = op_lin_repr + _non_root_lin_sep[0] + lr + _non_root_lin_sep[1]

    if f.is_root():
        sr = sr_operator + sr
    else:
        sr = sr_operator + r"\left(" + sr + r"\right)"

    f = f.__class__(
        ds,  # space
        sr,  # symbolic representation
        lr,
        False,
    )

    return f


def codifferential(f):
    """codifferential of a form."""
    ds = space_codifferential(f.space)

    lr = f._lin_repr
    sr = f._sym_repr

    op_lin_repr = _global_operator_lin_repr_setting['codifferential']
    sr_operator = _global_operator_sym_repr_setting['codifferential']

    if f.is_root():
        lr = op_lin_repr + lr
    else:
        lr = op_lin_repr + _non_root_lin_sep[0] + lr + _non_root_lin_sep[1]

    if f.is_root():
        sr = sr_operator + sr
    else:
        sr = sr_operator + r"\left(" + sr + r"\right)"

    f = f.__class__(
        ds,  # space
        sr,  # symbolic representation
        lr,
        False,
    )

    return f


def time_derivative(f, degree=1):
    """The time derivative operator."""
    if f.__class__.__name__ != 'Form':
        raise NotImplementedError(f"time_derivative on {f} is not implemented or even not possible at all.")
    else:
        pass

    lr = f._lin_repr
    sr = f._sym_repr

    if degree == 1:
        op_lin_repr = _global_operator_lin_repr_setting['time_derivative']
        sr_operator = _global_operator_sym_repr_setting['time_derivative']

        if f.is_root():
            lr = op_lin_repr + lr
        else:
            lr = op_lin_repr + _non_root_lin_sep[0] + lr + _non_root_lin_sep[1]

        if f.is_root():
            sr = sr_operator + sr
        else:
            sr = sr_operator + r"\left(" + sr + r"\right)"

        tdf = f.__class__(
            f.space,
            sr,
            lr,
            False,
        )

        return tdf

    else:
        raise NotImplementedError()


from src.spaces.operators import trace as space_trace


def trace(f):
    """The trace operator."""
    trf_space = space_trace(f.space)

    lr = f._lin_repr
    sr = f._sym_repr

    op_lin_repr = _global_operator_lin_repr_setting['trace']
    sr_operator = _global_operator_sym_repr_setting['trace']

    if f.is_root():
        lr = op_lin_repr + lr
    else:
        lr = op_lin_repr + _non_root_lin_sep[0] + lr + _non_root_lin_sep[1]

    if f.is_root():
        sr = sr_operator + sr
    else:
        sr = sr_operator + r"\left(" + sr + r"\right)"

    f = f.__class__(
        trf_space,  # space
        sr,  # symbolic representation
        lr,
        False,
    )

    return f


def cross_product(f1, f2):
    """f1 x f2"""
    s1 = f1.space
    s2 = f2.space

    cross_product_space = space_cross_product(s1, s2)

    lr_term1 = f1._lin_repr
    lr_term2 = f2._lin_repr
    lr_operator = _global_operator_lin_repr_setting['cross_product']

    sr_term1 = f1._sym_repr
    sr_term2 = f2._sym_repr
    sr_operator = _global_operator_sym_repr_setting['cross_product']

    if f1.is_root():
        pass
    else:
        lr_term1 = _non_root_lin_sep[0] + lr_term1 + _non_root_lin_sep[1]
        sr_term1 = r'\left(' + sr_term1 + r'\right)'
    if f2.is_root():
        pass
    else:
        lr_term2 = _non_root_lin_sep[0] + lr_term2 + _non_root_lin_sep[1]
        sr_term2 = r'\left(' + sr_term2 + r'\right)'
    lin_repr = lr_term1 + lr_operator + lr_term2
    sym_repr = sr_term1 + sr_operator + sr_term2

    f = f1.__class__(
        cross_product_space,  # space
        sym_repr,  # symbolic representation
        lin_repr,
        False
    )

    return f


def Cross_Product(f1, f2):
    r"""f1 X f2"""
    s1 = f1.space
    s2 = f2.space

    cross_product_space = space_Cross_Product(s1, s2)

    lr_term1 = f1._lin_repr
    lr_term2 = f2._lin_repr
    lr_operator = _global_operator_lin_repr_setting['Cross_Product']

    sr_term1 = f1._sym_repr
    sr_term2 = f2._sym_repr
    sr_operator = _global_operator_sym_repr_setting['Cross_Product']

    if f1.is_root():
        pass
    else:
        lr_term1 = _non_root_lin_sep[0] + lr_term1 + _non_root_lin_sep[1]
        sr_term1 = r'\left(' + sr_term1 + r'\right)'
    if f2.is_root():
        pass
    else:
        lr_term2 = _non_root_lin_sep[0] + lr_term2 + _non_root_lin_sep[1]
        sr_term2 = r'\left(' + sr_term2 + r'\right)'
    lin_repr = lr_term1 + lr_operator + lr_term2
    sym_repr = sr_term1 + sr_operator + sr_term2

    f = f1.__class__(
        cross_product_space,  # space
        sym_repr,  # symbolic representation
        lin_repr,
        False
    )

    return f



def convect(f1, f2):
    """f1.convect(f2)."""
    s1 = f1.space
    s2 = f2.space

    convect_space = space_convect(s1, s2)

    lr_term1 = f1._lin_repr
    lr_term2 = f2._lin_repr
    lr_operator = _global_operator_lin_repr_setting['convect']

    sr_term1 = f1._sym_repr
    sr_term2 = f2._sym_repr
    sr_operator = _global_operator_sym_repr_setting['convect']

    if f1.is_root():
        pass
    else:
        lr_term1 = _non_root_lin_sep[0] + lr_term1 + _non_root_lin_sep[1]
        sr_term1 = r'\left(' + sr_term1 + r'\right)'
    if f2.is_root():
        pass
    else:
        lr_term2 = _non_root_lin_sep[0] + lr_term2 + _non_root_lin_sep[1]
        sr_term2 = r'\left(' + sr_term2 + r'\right)'
    lin_repr = lr_term1 + lr_operator + lr_term2
    sym_repr = sr_term1 + sr_operator + sr_term2

    f = f1.__class__(
        convect_space,  # space
        sym_repr,  # symbolic representation
        lin_repr,
        False
    )

    return f


def tensor_product(f1, f2):
    """"""
    s1 = f1.space
    s2 = f2.space

    tensor_product_space = space_tensor_product(s1, s2)

    lr_term1 = f1._lin_repr
    lr_term2 = f2._lin_repr
    lr_operator = _global_operator_lin_repr_setting['tensor_product']

    sr_term1 = f1._sym_repr
    sr_term2 = f2._sym_repr
    sr_operator = _global_operator_sym_repr_setting['tensor_product']

    if f1.is_root():
        pass
    else:
        lr_term1 = _non_root_lin_sep[0] + lr_term1 + _non_root_lin_sep[1]
        sr_term1 = r'\left(' + sr_term1 + r'\right)'
    if f2.is_root():
        pass
    else:
        lr_term2 = _non_root_lin_sep[0] + lr_term2 + _non_root_lin_sep[1]
        sr_term2 = r'\left(' + sr_term2 + r'\right)'
    lin_repr = lr_term1 + lr_operator + lr_term2
    sym_repr = sr_term1 + sr_operator + sr_term2

    f = f1.__class__(
        tensor_product_space,  # space
        sym_repr,  # symbolic representation
        lin_repr,
        False
    )

    return f
