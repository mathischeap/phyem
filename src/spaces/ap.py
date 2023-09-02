# -*- coding: utf-8 -*-
"""
Algebraic Proxy.
"""
from src.spaces.main import *
from src.config import _form_evaluate_at_repr_setting
from src.spaces.main import _default_space_degree_repr
from src.spaces.main import _degree_str_maker
from src.algebra.array import _root_array
from src.algebra.nonlinear_operator import AbstractNonlinearOperator
from src.spaces.operators import d


__all__ = [
    '_VarPar_M',
    '_VarPar_E',

    '_VarPar_boundary_dp_vector',

    '_VarPar_astA_x_astB_ip_tC',
    '_VarPar_astA_x_B_ip_tC',
    '_VarPar_A_x_astB_ip_tC',
    '_VarPar_A_x_B_ip_C',           # nonlinear

    '_VarPar_dastA_astA_tp_tC',
    '_VarPar_dastA_tB_tp_astA',
    '_VarPar_dtA_astB_tp_astB',
    '_VarPar_dA_B_tp_C__1Known',
    '_VarPar_dA_B_tp_C__2Known',
    '_VarPar_dA_B_tp_C',            # nonlinear

    '_VarPar_A_B_tp_C__1Known',
    '_VarPar_A_B_tp_C__2Known',
    '_VarPar_A_B_tp_C',            # nonlinear

    '_VarPar_l2_inner_product_db_bf',
]


# --- basic ---------------------------------------------------------------------------------------

def _VarPar_M(s0, s1, d0, d1):
    """parse l2 inner product mass matrix."""
    assert s0 == s1, f"spaces do not match."

    sym, lin = _VarSetting_mass_matrix
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


def _VarPar_E(f, transpose=False):
    """"""
    s = f.space
    degree = f._degree
    assert degree is not None, f"space is not finite."

    ds = d(s)
    degree = _degree_str_maker(degree)

    if transpose:
        sym, lin = _VarSetting_d_matrix_transpose
        lin = lin.replace('{space_pure_lin_repr}', str(s._pure_lin_repr))
        lin = lin.replace('{d}', degree)
        sym += r"^{" + str((s.k, s.k+1)) + r"}"
        shape = (f._ap_shape(), ds._sym_repr + _default_space_degree_repr + degree)

    else:
        sym, lin = _VarSetting_d_matrix
        lin = lin.replace('{space_pure_lin_repr}', str(s._pure_lin_repr))
        lin = lin.replace('{d}', degree)
        sym += r"^{" + str((s.k+1, s.k)) + r"}"
        shape = (ds._sym_repr + _default_space_degree_repr + degree, f._ap_shape())

    D = _root_array(sym, lin, shape)

    return D


# --- natural bc ----------------------------------------------------------------------------------

def _VarPar_boundary_dp_vector(rf0, f1):
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
    sym, lin = _VarSetting_boundary_dp_vector[:2]
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


# (w x u, u) --------------------------------------------------------------------------------------

def _VarPar_astA_x_astB_ip_tC(gA, gB, tC):
    """"""
    sym, lin = _VarSetting_astA_x_astB_ip_tC[:2]

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


def _VarPar_astA_x_B_ip_tC(gA, B, tC):
    """<A x B, C> where A is given (ast). and C is the test form."""
    sym, lin = _VarSetting_astA_x_B_ip_tC[:2]

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


def _VarPar_A_x_astB_ip_tC(A, gB, tC):
    """"""
    sym, lin = _VarSetting_A_x_astB_ip_tC[:2]

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


def _VarPar_A_x_B_ip_C(A, B, C):
    """(AxB, C)"""
    sym, lin = _VarSetting_A_x_B_ip_C[:2]

    sym += rf"\left({A._sym_repr}, {B._sym_repr}, {C._sym_repr}\right)"
    lin = lin.replace('{A}', A._pure_lin_repr)
    lin = lin.replace('{B}', B._pure_lin_repr)
    lin = lin.replace('{C}', C._pure_lin_repr)

    mda = AbstractNonlinearOperator(sym, lin)
    return mda


# -----(dA, B otimes C) ----------------------------------------------------------------------------
def _VarPar_dastA_astA_tp_tC(gA, tC):
    """<d(gA), gA otimes tC>"""
    sym, lin = _VarSetting_dastA_astA_tp_tC[:2]

    sym += r"_{(" + gA._sym_repr + ',' + gA._sym_repr + r")}"
    lin = lin.replace('{A}', gA._pure_lin_repr)
    lin = lin.replace('{C}', tC._pure_lin_repr)

    s0 = tC.space
    str_d0 = _degree_str_maker(tC._degree)
    shape0 = s0._sym_repr + _default_space_degree_repr + str_d0

    shape = (shape0, 1)
    ra = _root_array(sym, lin, shape)
    return ra


def _VarPar_dastA_tB_tp_astA(gA, tB):
    """<d(gA), tB otimes gA>"""
    sym, lin = _VarSetting_dastA_tB_tp_astA[:2]

    sym += r"_{(" + gA._sym_repr + ',' + gA._sym_repr + r")}"
    lin = lin.replace('{A}', gA._pure_lin_repr)
    lin = lin.replace('{B}', tB._pure_lin_repr)

    s0 = tB.space
    str_d0 = _degree_str_maker(tB._degree)
    shape0 = s0._sym_repr + _default_space_degree_repr + str_d0

    shape = (shape0, 1)
    ra = _root_array(sym, lin, shape)
    return ra


def _VarPar_dtA_astB_tp_astB(tA, gB):
    """<dtA, gB otimes gB>"""
    sym, lin = _VarSetting_dtA_astB_tp_astB[:2]

    sym += r"_{(" + gB._sym_repr + ',' + gB._sym_repr + r")}"
    lin = lin.replace('{A}', tA._pure_lin_repr)
    lin = lin.replace('{B}', gB._pure_lin_repr)

    s0 = tA.space
    str_d0 = _degree_str_maker(tA._degree)
    shape0 = s0._sym_repr + _default_space_degree_repr + str_d0

    shape = (shape0, 1)
    ra = _root_array(sym, lin, shape)
    return ra


def _VarPar_dA_B_tp_C__1Known(A, B, C, kf, tf):
    """(dA, B otimes C), one off A, B, C is known, one of the other two is the test form."""
    assert (A is not B) and (A is not C) and (B is not C), f"A, B, C must be different."
    assert tf in (A, B, C) and tf is not kf, \
        f'must be. The test form must be among A, B, C and is not the know form'
    unknown = None
    for _ in (A, B, C):
        if _ is tf or _ is kf:
            pass
        else:
            assert unknown is None, f"must find only one unknown form."
            unknown = _
    assert unknown is not None, f"must find a unknown form."

    tf_index, kf_index, unknown_index = None, None, None
    for index, _ in enumerate((A, B, C)):
        if _ is tf:
            tf_index = index
        elif _ is kf:
            kf_index = index
        else:
            unknown_index = index
    assert tf_index is not None and kf_index is not None and unknown_index is not None, f"must be!"
    sym, lin = _VarSetting_dA_B_tp_C__1Known[:2]
    add_sym_list = ['', '', '']
    add_sym_list[tf_index] = tf._sym_repr
    add_sym_list[kf_index] = kf._sym_repr
    add_sym_list[unknown_index] = unknown._sym_repr
    add_sym_list = ','.join(add_sym_list)
    sym += r"_{(" + add_sym_list + r")}"
    lin = lin.replace('{A}', A._pure_lin_repr)
    lin = lin.replace('{B}', B._pure_lin_repr)
    lin = lin.replace('{C}', C._pure_lin_repr)
    lin = lin.replace('{K}', kf._pure_lin_repr)
    lin = lin.replace('{T}', tf._pure_lin_repr)
    lin = lin.replace('{U}', unknown._pure_lin_repr)

    s0 = tf.space
    s1 = unknown.space
    d0 = tf._degree
    d1 = unknown._degree
    str_d0 = _degree_str_maker(d0)
    str_d1 = _degree_str_maker(d1)

    shape0 = s0._sym_repr + _default_space_degree_repr + str_d0
    shape1 = s1._sym_repr + _default_space_degree_repr + str_d1

    shape = (shape0, shape1)
    ra = _root_array(sym, lin, shape)
    return ra


def _VarPar_dA_B_tp_C__2Known(A, B, C, kf1, kf2, tf):
    """A,B,C are all different. (dA, B otimes C), two of A, B, C are known, the rest one is the test form."""
    assert (A is not B) and (A is not C) and (B is not C), f"A, B, C must be different."
    assert (tf in (A, B, C)) and (tf is not kf1) and (tf is not kf2) and (kf1 is not kf2), \
        f'A,B,C are all different. (dA, B otimes C), two of A, B, C are known, the rest one is the test form.'
    assert (kf1 in (A, B, C)) and (kf2 in (A, B, C)), \
        f'A,B,C are all different. (dA, B otimes C), two of A, B, C are known, the rest one is the test form.'
    tf_index, kf1_index, kf2_index = None, None, None
    for index, _ in enumerate((A, B, C)):
        if _ is tf:
            tf_index = index
        elif _ is kf1:
            kf1_index = index
        elif _ is kf2:
            kf2_index = index
        else:
            raise Exception()
    assert tf_index is not None and kf1_index is not None and kf2_index is not None, f"must be!"
    sym, lin = _VarSetting_dA_B_tp_C__2Known[:2]
    add_sym_list = ['', '', '']
    add_sym_list[tf_index] = tf._sym_repr
    add_sym_list[kf1_index] = kf1._sym_repr
    add_sym_list[kf2_index] = kf2._sym_repr
    add_sym_list = ','.join(add_sym_list)
    sym += r"_{(" + add_sym_list + r")}"
    lin = lin.replace('{A}', A._pure_lin_repr)
    lin = lin.replace('{B}', B._pure_lin_repr)
    lin = lin.replace('{C}', C._pure_lin_repr)
    lin = lin.replace('{K1}', kf1._pure_lin_repr)
    lin = lin.replace('{K2}', kf2._pure_lin_repr)
    lin = lin.replace('{T}', tf._pure_lin_repr)

    s0 = tf.space
    str_d0 = _degree_str_maker(tf._degree)
    shape0 = s0._sym_repr + _default_space_degree_repr + str_d0

    shape = (shape0, 1)
    ra = _root_array(sym, lin, shape)
    return ra


def _VarPar_dA_B_tp_C(A, B, C):
    """(dA, B otimes C), nonlinear"""
    assert A is not B and A is not C and B is not C, f"A, B, C must be different."
    sym, lin = _VarSetting_dA_B_tp_C[:2]

    sym += rf"\left({A._sym_repr}, {B._sym_repr}, {C._sym_repr}\right)"
    lin = lin.replace('{A}', A._pure_lin_repr)
    lin = lin.replace('{B}', B._pure_lin_repr)
    lin = lin.replace('{C}', C._pure_lin_repr)

    mda = AbstractNonlinearOperator(sym, lin)
    return mda


# (A, B otimes C) --------------------------------------------------------------------------------

def _VarPar_A_B_tp_C__1Known(A, B, C, kf, tf):
    """(dA, B otimes C), one off A, B, C is known, one of the other two is the test form."""
    assert (A is not B) and (A is not C) and (B is not C), f"A, B, C must be different."
    assert tf in (A, B, C) and tf is not kf, \
        f'must be. The test form must be among A, B, C and is not the know form'
    unknown = None
    for _ in (A, B, C):
        if _ is tf or _ is kf:
            pass
        else:
            assert unknown is None, f"must find only one unknown form."
            unknown = _
    assert unknown is not None, f"must find a unknown form."

    tf_index, kf_index, unknown_index = None, None, None
    for index, _ in enumerate((A, B, C)):
        if _ is tf:
            tf_index = index
        elif _ is kf:
            kf_index = index
        else:
            unknown_index = index
    assert tf_index is not None and kf_index is not None and unknown_index is not None, f"must be!"
    sym, lin = _VarSetting_A_B_tp_C__1Known[:2]
    add_sym_list = ['', '', '']
    add_sym_list[tf_index] = tf._sym_repr
    add_sym_list[kf_index] = kf._sym_repr
    add_sym_list[unknown_index] = unknown._sym_repr
    add_sym_list = ','.join(add_sym_list)
    sym += r"_{(" + add_sym_list + r")}"
    lin = lin.replace('{A}', A._pure_lin_repr)
    lin = lin.replace('{B}', B._pure_lin_repr)
    lin = lin.replace('{C}', C._pure_lin_repr)
    lin = lin.replace('{K}', kf._pure_lin_repr)
    lin = lin.replace('{T}', tf._pure_lin_repr)
    lin = lin.replace('{U}', unknown._pure_lin_repr)

    s0 = tf.space
    s1 = unknown.space
    d0 = tf._degree
    d1 = unknown._degree
    str_d0 = _degree_str_maker(d0)
    str_d1 = _degree_str_maker(d1)

    shape0 = s0._sym_repr + _default_space_degree_repr + str_d0
    shape1 = s1._sym_repr + _default_space_degree_repr + str_d1

    shape = (shape0, shape1)
    ra = _root_array(sym, lin, shape)
    return ra


def _VarPar_A_B_tp_C__2Known(A, B, C, kf1, kf2, tf):
    """A,B,C are all different. (dA, B otimes C), two of A, B, C are known, the rest one is the test form."""
    assert (A is not B) and (A is not C) and (B is not C), f"A, B, C must be different."
    assert (tf in (A, B, C)) and (tf is not kf1) and (tf is not kf2) and (kf1 is not kf2), \
        f'A,B,C are all different. (dA, B otimes C), two of A, B, C are known, the rest one is the test form.'
    assert (kf1 in (A, B, C)) and (kf2 in (A, B, C)), \
        f'A,B,C are all different. (dA, B otimes C), two of A, B, C are known, the rest one is the test form.'
    tf_index, kf1_index, kf2_index = None, None, None
    for index, _ in enumerate((A, B, C)):
        if _ is tf:
            tf_index = index
        elif _ is kf1:
            kf1_index = index
        elif _ is kf2:
            kf2_index = index
        else:
            raise Exception()
    assert tf_index is not None and kf1_index is not None and kf2_index is not None, f"must be!"
    sym, lin = _VarSetting_A_B_tp_C__2Known[:2]
    add_sym_list = ['', '', '']
    add_sym_list[tf_index] = tf._sym_repr
    add_sym_list[kf1_index] = kf1._sym_repr
    add_sym_list[kf2_index] = kf2._sym_repr
    add_sym_list = ','.join(add_sym_list)
    sym += r"_{(" + add_sym_list + r")}"
    lin = lin.replace('{A}', A._pure_lin_repr)
    lin = lin.replace('{B}', B._pure_lin_repr)
    lin = lin.replace('{C}', C._pure_lin_repr)
    lin = lin.replace('{K1}', kf1._pure_lin_repr)
    lin = lin.replace('{K2}', kf2._pure_lin_repr)
    lin = lin.replace('{T}', tf._pure_lin_repr)

    s0 = tf.space
    str_d0 = _degree_str_maker(tf._degree)
    shape0 = s0._sym_repr + _default_space_degree_repr + str_d0

    shape = (shape0, 1)
    ra = _root_array(sym, lin, shape)
    return ra


def _VarPar_A_B_tp_C(A, B, C):
    """(dA, B otimes C), nonlinear"""
    assert A is not B and A is not C and B is not C, f"A, B, C must be different."
    sym, lin = _VarSetting_A_B_tp_C[:2]

    sym += rf"\left({A._sym_repr}, {B._sym_repr}, {C._sym_repr}\right)"
    lin = lin.replace('{A}', A._pure_lin_repr)
    lin = lin.replace('{B}', B._pure_lin_repr)
    lin = lin.replace('{C}', C._pure_lin_repr)

    mda = AbstractNonlinearOperator(sym, lin)
    return mda


# --- (bundle form, special diagonal bundle form)----------------------------------------------------
def _VarPar_l2_inner_product_db_bf(db, bf, transpose=False):
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
        sym, lin = _VarSetting_IP_matrix_bf_db

    else:
        sym, lin = _VarSetting_IP_matrix_db_bf

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
