# -*- coding: utf-8 -*-
"""
Algebraic Proxy.

pH-lib@RAM-EEMCS-UT
created at: 3/20/2023 5:17 PM
"""
import sys

if './' not in sys.path:
    sys.path.append('./')

from src.spaces.main import _default_mass_matrix_reprs
from src.spaces.main import _default_d_matrix_reprs, _default_d_matrix_transpose_reprs
from src.spaces.main import _default_wedge_vector_repr
from src.spaces.main import _default_trace_matrix_repr
from src.spaces.main import _default_space_degree_repr
from src.algebra.array import _root_array
from src.spaces.operators import d, trace


def _parse_l2_inner_product_mass_matrix(s0, s1, d0, d1):
    """parse l2 inner product mass matrix."""
    assert s0 == s1, f"spaces do not match."

    if s0.__class__.__name__ == 'ScalarValuedFormSpace':
        sym, lin = _default_mass_matrix_reprs['Lambda']
        assert d0 is not None and d1 is not None, f"space is not finite."
        sym += rf"^{s0.k}"

        lin = lin.replace('{n}', str(s0.n))
        lin = lin.replace('{k}', str(s0.k))
        lin = lin.replace('{(d0,d1)}', str((d0, d1)))

        return _root_array(
            sym, lin, (
                s0._sym_repr + _default_space_degree_repr + str(d0),
                s1._sym_repr + _default_space_degree_repr + str(d1)
            ), symmetric=True,
        )

    else:
        raise NotImplementedError()


def _parse_d_matrix(f, transpose=False):
    """"""
    s = f.space
    degree = f._degree
    if s.__class__.__name__ == 'ScalarValuedFormSpace':
        assert degree is not None, f"space is not finite."

        ds = d(s)

        if transpose:
            sym, lin = _default_d_matrix_transpose_reprs['Lambda']
            lin = lin.replace('{n}', str(s.n))
            lin = lin.replace('{k}', str(s.k))
            lin = lin.replace('{d}', str(degree))
            sym += r"^{" + str((s.k, s.k+1)) + r"}"
            shape = (f._ap_shape(), ds._sym_repr + _default_space_degree_repr + str(degree))

        else:
            sym, lin = _default_d_matrix_reprs['Lambda']
            lin = lin.replace('{n}', str(s.n))
            lin = lin.replace('{k}', str(s.k))
            lin = lin.replace('{d}', str(degree))
            sym += r"^{" + str((s.k+1, s.k)) + r"}"
            shape = (ds._sym_repr + _default_space_degree_repr + str(degree), f._ap_shape())
        D = _root_array(sym, lin, shape)

        return D

    else:
        raise NotImplementedError()


def _parse_wedge_vector(rf0, s1, d1):
    """

    Parameters
    ----------
    rf0 :
        It is root f0 dependent. So do not use s0.
    s1
    d1

    Returns
    -------

    """
    s0 = rf0.space
    if s0.__class__.__name__ == 'ScalarValuedFormSpace':
        assert d1 is not None, f"space is not finite."
        sym, lin = _default_wedge_vector_repr['Lambda']
        lin = lin.replace('{f0}', rf0._pure_lin_repr)
        lin = lin.replace('{d}', str(d1))

        sym += rf"_{s0.k}"

        return _root_array(sym, lin, (s1._sym_repr + _default_space_degree_repr + str(d1), 1))

    else:
        raise NotImplementedError()


def _parse_trace_matrix(f):
    """"""
    s = f.space
    degree = f._degree
    if s.__class__.__name__ == 'ScalarValuedFormSpace':
        sym, lin = _default_trace_matrix_repr['Lambda']
        lin = lin.replace('{n}', str(s.n))
        lin = lin.replace('{k}', str(s.k))
        lin = lin.replace('{d}', str(degree))
        sym += rf'_{s.k}'
        trace_space = trace(s)
        return _root_array(sym, lin, (trace_space._sym_repr + _default_space_degree_repr + str(degree), f._ap_shape()))

    else:
        raise NotImplementedError()
