# -*- coding: utf-8 -*-
"""
Algebraic Proxy.

pH-lib@RAM-EEMCS-UT
created at: 3/20/2023 5:17 PM
"""

from src.spaces.main import _default_mass_matrix_reprs
from src.spaces.main import _default_d_matrix_reprs, _default_d_matrix_transpose_reprs
from src.spaces.main import _default_boundary_dp_vector_repr

from src.spaces.main import _default_space_degree_repr

from src.spaces.main import _degree_str_maker

from src.algebra.array import _root_array
from src.spaces.operators import d


def _parse_l2_inner_product_mass_matrix(s0, s1, d0, d1):
    """parse l2 inner product mass matrix."""
    assert s0 == s1, f"spaces do not match."

    sym, lin = _default_mass_matrix_reprs
    assert d0 is not None and d1 is not None, f"space is not finite."
    sym += rf"^{s0.k}"

    lin = lin.replace('{space_pure_lin_repr}', str(s0._pure_lin_repr))

    str_d0 = _degree_str_maker(d0)
    str_d1 = _degree_str_maker(d1)

    lin = lin.replace('{d0}', str_d0)
    lin = lin.replace('{d1}', str_d1)

    return _root_array(
        sym, lin, (
            s0._sym_repr + _default_space_degree_repr + str_d0,
            s1._sym_repr + _default_space_degree_repr + str_d1
        ), symmetric=True,
    )


def _parse_d_matrix(f, transpose=False):
    """"""
    s = f.space
    degree = f._degree
    assert degree is not None, f"space is not finite."

    ds = d(s)
    degree = _degree_str_maker(degree)

    if transpose:
        sym, lin = _default_d_matrix_transpose_reprs
        lin = lin.replace('{space_pure_lin_repr}', str(s._pure_lin_repr))
        lin = lin.replace('{d}', degree)
        sym += r"^{" + str((s.k, s.k+1)) + r"}"
        shape = (f._ap_shape(), ds._sym_repr + _default_space_degree_repr + degree)

    else:
        sym, lin = _default_d_matrix_reprs
        lin = lin.replace('{space_pure_lin_repr}', str(s._pure_lin_repr))
        lin = lin.replace('{d}', degree)
        sym += r"^{" + str((s.k+1, s.k)) + r"}"
        shape = (ds._sym_repr + _default_space_degree_repr + degree, f._ap_shape())

    D = _root_array(sym, lin, shape)

    return D


def _parse_boundary_dp_vector(rf0, s1, d1):
    """
    <tr star rf0, tr v1> where v1 in s1, and s1 is of degree d1


    Parameters
    ----------
    rf0 :
        It is root-f0-dependent. So do not use s0.
    s1
    d1

    Returns
    -------

    """
    assert d1 is not None, f"space is not finite."
    sym, lin = _default_boundary_dp_vector_repr
    lin = lin.replace('{f0}', rf0._pure_lin_repr)
    lin = lin.replace('{s1}', s1._pure_lin_repr)
    d1 = _degree_str_maker(d1)
    lin = lin.replace('{d}', d1)
    sym += rf"_{s1.k}"
    return _root_array(sym, lin, (s1._sym_repr + _default_space_degree_repr + d1, 1))
