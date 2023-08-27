# -*- coding: utf-8 -*-
"""
Algebraic Proxy.
"""
from src.spaces.main import _default_mass_matrix_reprs
from src.spaces.main import _default_astA_x_astB_ip_tC_reprs
from src.spaces.main import _default_astA_x_B_ip_tC_reprs, _default_astA_tp_B_tC_reprs
from src.spaces.main import _default_A_x_astB_ip_tC_reprs, _default_dastA_astA_tp_tC_reprs
from src.spaces.main import _default_A_x_B_ip_C_reprs, _default_A_tp_A_ip_C_reprs, _default_dA_dA_tp_C_reprs
from src.spaces.main import _default_d_matrix_reprs, _default_d_matrix_transpose_reprs
from src.spaces.main import _default_boundary_dp_vector_reprs, _default_dA_astB_tp_tC_reprs
from src.spaces.main import _default_dastA_B_tp_tC_reprs
from src.spaces.main import _default_mass_matrix_db_bf_reprs, _default_mass_matrix_bf_db_reprs

from src.config import _form_evaluate_at_repr_setting

from src.spaces.main import _default_space_degree_repr

from src.spaces.main import _degree_str_maker

from src.algebra.array import _root_array
from src.algebra.nonlinear_operator import AbstractNonlinearOperator
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


def _parse_l2_inner_product_db_bf(db, bf, transpose=False):
    """
    if ``transpose``, 0-axis of the output refers to `bd`, else, 0-axis refers to `df`.

    Parameters
    ----------
    db
    bf
    transpose

    Returns
    -------

    """
    db_space = db.space
    bf_space = bf.space
    if transpose:
        sym, lin = _default_mass_matrix_bf_db_reprs

    else:
        sym, lin = _default_mass_matrix_db_bf_reprs

    lin = lin.replace('{db_space_pure_lin_repr}', str(db_space._pure_lin_repr))
    lin = lin.replace('{bf_space_pure_lin_repr}', str(bf_space._pure_lin_repr))

    degree_db = db._degree
    degree_bf = bf._degree

    str_db = _degree_str_maker(degree_db)
    str_bf = _degree_str_maker(degree_bf)

    lin = lin.replace('{degree_db}', str_db)
    lin = lin.replace('{degree_bf}', str_bf)

    if transpose:
        return _root_array(
            sym, lin, (
                bf_space._sym_repr + _default_space_degree_repr + str_bf,
                db_space._sym_repr + _default_space_degree_repr + str_db
            ), symmetric=False,
        )
    else:
        return _root_array(
            sym, lin, (
                db_space._sym_repr + _default_space_degree_repr + str_db,
                bf_space._sym_repr + _default_space_degree_repr + str_bf
            ), symmetric=False,
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


def _parse_boundary_dp_vector(rf0, f1):
    """
    <tr star rf0, tr f1> where f1


    Parameters
    ----------
    rf0 :
        It is root-f0-dependent. So do not use s0.
    f1

    Returns
    -------

    """
    s1 = f1.space
    d1 = f1._degree
    assert d1 is not None, f"space is not finite."
    sym, lin = _default_boundary_dp_vector_reprs[:2]
    lin = lin.replace('{f0}', rf0._pure_lin_repr)
    lin = lin.replace('{f1}', f1._pure_lin_repr)
    d1 = _degree_str_maker(d1)
    lin = lin.replace('{d}', d1)
    sym += rf"_{s1.k}"
    if _form_evaluate_at_repr_setting['lin'] in rf0._pure_lin_repr:
        evaluation_sym = _form_evaluate_at_repr_setting['sym']
        rf0_sym_repr = rf0._sym_repr
        rf0_superscript = rf0_sym_repr.split(evaluation_sym[1])[1]
        rf0_superscript = rf0_superscript[:-len(evaluation_sym[2])]
        sym = sym + r"^{(" + rf0_superscript + r")}"
    else:
        pass
    ra = _root_array(sym, lin, (s1._sym_repr + _default_space_degree_repr + d1, 1))
    return ra


def _parse_astA_x_astB_ip_tC(gA, gB, tC):
    """"""
    sym, lin = _default_astA_x_astB_ip_tC_reprs[:2]

    sym += r"_{(" + gA._sym_repr + ',' + gB._sym_repr + r")}"
    lin = lin.replace('{A}', gA._pure_lin_repr)
    lin = lin.replace('{B}', gB._pure_lin_repr)
    lin = lin.replace('{C}', tC._pure_lin_repr)

    s0 = tC.space
    d0 = tC._degree
    str_d0 = _degree_str_maker(d0)

    shape0 = s0._sym_repr + _default_space_degree_repr + str_d0

    shape = (shape0, 1)
    ra = _root_array(sym, lin, shape)
    return ra


def _parse_astA_x_B_ip_tC(gA, B, tC):
    """<A x B, C> where A is given (ast). and C is the test form."""
    sym, lin = _default_astA_x_B_ip_tC_reprs[:2]

    sym += r"_{" + gA._sym_repr + r"}"
    lin = lin.replace('{A}', gA._pure_lin_repr)
    lin = lin.replace('{B}', B._pure_lin_repr)
    lin = lin.replace('{C}', tC._pure_lin_repr)

    s0 = tC.space
    s1 = B.space
    d0 = tC._degree
    d1 = B._degree
    str_d0 = _degree_str_maker(d0)
    str_d1 = _degree_str_maker(d1)

    shape0 = s0._sym_repr + _default_space_degree_repr + str_d0
    shape1 = s1._sym_repr + _default_space_degree_repr + str_d1

    shape = (shape0, shape1)
    ra = _root_array(sym, lin, shape)
    return ra


def _parse_A_x_astB_ip_tC(A, gB, tC):
    """"""
    sym, lin = _default_A_x_astB_ip_tC_reprs[:2]

    sym += r"_{" + gB._sym_repr + r"}"
    lin = lin.replace('{A}', A._pure_lin_repr)
    lin = lin.replace('{B}', gB._pure_lin_repr)
    lin = lin.replace('{C}', tC._pure_lin_repr)

    s0 = tC.space
    s1 = A.space
    d0 = tC._degree
    d1 = A._degree
    str_d0 = _degree_str_maker(d0)
    str_d1 = _degree_str_maker(d1)

    shape0 = s0._sym_repr + _default_space_degree_repr + str_d0
    shape1 = s1._sym_repr + _default_space_degree_repr + str_d1

    shape = (shape0, shape1)
    ra = _root_array(sym, lin, shape)
    return ra


def _parse_dA_astB_tp_tC(A, gB, tC):
    """"""

    sym, lin = _default_dA_astB_tp_tC_reprs[:2]

    sym += r"_{" + gB._sym_repr + r"}"
    lin = lin.replace('{A}', A._pure_lin_repr)
    lin = lin.replace('{B}', gB._pure_lin_repr)
    lin = lin.replace('{C}', tC._pure_lin_repr)

    s0 = tC.space
    s1 = A.space
    d0 = tC._degree
    d1 = A._degree
    str_d0 = _degree_str_maker(d0)
    str_d1 = _degree_str_maker(d1)

    shape0 = s0._sym_repr + _default_space_degree_repr + str_d0
    shape1 = s1._sym_repr + _default_space_degree_repr + str_d1

    shape = (shape0, shape1)
    ra = _root_array(sym, lin, shape)
    return ra


def _parse_dastA_B_tp_tC(gA, B, tC):
    """"""

    sym, lin = _default_dastA_B_tp_tC_reprs[:2]

    sym += r"_{" + gA._sym_repr + r"}"
    lin = lin.replace('{A}', gA._pure_lin_repr)
    lin = lin.replace('{B}', B._pure_lin_repr)
    lin = lin.replace('{C}', tC._pure_lin_repr)

    s0 = tC.space
    s1 = B.space
    d0 = tC._degree
    d1 = B._degree
    str_d0 = _degree_str_maker(d0)
    str_d1 = _degree_str_maker(d1)

    shape0 = s0._sym_repr + _default_space_degree_repr + str_d0
    shape1 = s1._sym_repr + _default_space_degree_repr + str_d1

    shape = (shape0, shape1)
    ra = _root_array(sym, lin, shape)
    return ra


def _parse_astA_tp_B_tC(gA, B, tC):
    """"""

    sym, lin = _default_astA_tp_B_tC_reprs[:2]

    sym += r"_{" + gA._sym_repr + r"}"
    lin = lin.replace('{A}', gA._pure_lin_repr)
    lin = lin.replace('{B}', B._pure_lin_repr)
    lin = lin.replace('{C}', tC._pure_lin_repr)

    s0 = tC.space
    s1 = B.space
    d0 = tC._degree
    d1 = B._degree
    str_d0 = _degree_str_maker(d0)
    str_d1 = _degree_str_maker(d1)

    shape0 = s0._sym_repr + _default_space_degree_repr + str_d0
    shape1 = s1._sym_repr + _default_space_degree_repr + str_d1

    shape = (shape0, shape1)
    ra = _root_array(sym, lin, shape)
    return ra


def _parse_A_x_B_ip_C(A, B, C):
    """"""
    sym, lin = _default_A_x_B_ip_C_reprs[:2]

    sym += rf"\left({A._sym_repr}, {B._sym_repr}, {C._sym_repr}\right)"
    lin = lin.replace('{A}', A._pure_lin_repr)
    lin = lin.replace('{B}', B._pure_lin_repr)
    lin = lin.replace('{C}', C._pure_lin_repr)

    mda = AbstractNonlinearOperator(sym, lin)
    return mda


def _parse_fa_tp_fa__ip__f1(A, C):
    """(a otimes a, c)"""
    sym, lin = _default_A_tp_A_ip_C_reprs[:2]

    sym += rf"\left({A._sym_repr}, {A._sym_repr}, {C._sym_repr}\right)"
    lin = lin.replace('{A}', A._pure_lin_repr)
    lin = lin.replace('{C}', C._pure_lin_repr)
    mda = AbstractNonlinearOperator(sym, lin)
    return mda


def _parse_dA_dA_tp_C(A, C):
    """(dA, A otimes c)"""
    sym, lin = _default_dA_dA_tp_C_reprs[:2]

    sym += rf"\left({A._sym_repr}, {A._sym_repr}, {C._sym_repr}\right)"
    lin = lin.replace('{A}', A._pure_lin_repr)
    lin = lin.replace('{C}', C._pure_lin_repr)

    mda = AbstractNonlinearOperator(sym, lin)
    return mda


def _parse_dastA_astA_tp_C(gA, tC):
    """(dgA, gA otimes c)"""
    sym, lin = _default_dastA_astA_tp_tC_reprs[:2]

    sym += r"_{(" + gA._sym_repr + ',' + gA._sym_repr + r")}"
    lin = lin.replace('{A}', gA._pure_lin_repr)
    lin = lin.replace('{C}', tC._pure_lin_repr)

    s0 = tC.space
    d0 = tC._degree
    str_d0 = _degree_str_maker(d0)

    shape0 = s0._sym_repr + _default_space_degree_repr + str_d0

    shape = (shape0, 1)
    ra = _root_array(sym, lin, shape)
    return ra
