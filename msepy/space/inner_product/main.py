# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from src.config import _setting
from tools.frozen import Frozen
from tools.quadrature import Quadrature
from scipy.sparse import csr_matrix


def _find_dim(m):
    """"""
    d = list()
    while 1:
        d.append(len(m))
        if isinstance(m[0], np.ndarray):
            break
        else:
            m = m[0]
    return d


def _make_quad(space, degree):
    """"""
    indicator = space.abstract.indicator
    is_linear = space.mesh.elements._is_linear()
    if indicator == 'bundle':
        _P_ = space[degree].p

        quad_degrees = list()
        quad_types = list()
        for _pi_ in _P_:

            if is_linear:  # ALL elements are linear.
                high_accuracy = _setting['high_accuracy']
                if high_accuracy:
                    quad_degree = [p + 1 for p in _pi_]
                    quad_type = 'Gauss'
                    # +1 for conservation
                else:
                    quad_degree = [p for p in _pi_]
                    # + 0 for much sparser matrices.
                    quad_type = space[degree].ntype
            else:
                quad_degree = [p + 2 for p in _pi_]
                quad_type = 'Gauss'

            quad_degrees.append(
                quad_degree
            )
            quad_types.append(
                quad_type
            )

        quad_degrees = np.array(quad_degrees)
        quad_degree = np.max(quad_degrees, axis=0)

        if all([_ == quad_types[0] for _ in quad_types]):
            quad_type = quad_types[0]
        else:
            quad_type = 'Gauss'

        quad = (quad_degree, quad_type)

    elif indicator in ('Lambda', 'bundle-diagonal'):
        if is_linear:  # ALL elements are linear.
            high_accuracy = _setting['high_accuracy']
            if high_accuracy:
                quad_degree = [p + 1 for p in space[degree].p]
                # +1 for conservation
                quad_type = 'Gauss'
            else:
                quad_degree = [p for p in space[degree].p]
                # + 0 for much sparser matrices.
                quad_type = space[degree].ntype
            quad = (quad_degree, quad_type)
        else:
            quad_degree = [p + 2 for p in space[degree].p]
            quad = (quad_degree, 'Gauss')
    else:
        raise NotImplementedError()

    return quad


class MsePyInnerProduct(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._freeze()

    def __call__(self, degree, other_space_at_degree, special_key=None):
        """"""
        space0 = self._space
        degree0 = degree
        space1 = other_space_at_degree._space
        degree1 = other_space_at_degree._degree
        mesh0, mesh1 = space0.mesh, space1.mesh
        assert mesh0 is mesh1, f"mesh does not match."
        M_reference_element_dict = _make_inner_product_matrix(
            space0, space1, degree0, degree1, special_key
        )
        return mesh0.elements._index_mapping.distribute_according_to_reference_elements_dict(
            M_reference_element_dict
        )


def _make_inner_product_matrix(space0, space1, degree0, degree1, special_key):
    """"""
    quad0 = _make_quad(space0, degree0)
    quad1 = _make_quad(space1, degree1)
    qd0, qt0 = quad0
    qd1, qt1 = quad1

    quad_degree = np.array((qd0, qd1))
    quad_degree = np.max(quad_degree, axis=0)

    if qt0 == qt1:
        quad_type = qt0
    else:
        quad_type = 'Gauss'

    quad = Quadrature(quad_degree, category=quad_type)
    quad_nodes = quad.quad_nodes
    quad_weights = quad.quad_weights_ravel

    rcM0 = space0.reconstruction_matrix(degree0, *quad_nodes)
    rcM1 = space1.reconstruction_matrix(degree1, *quad_nodes)

    mesh = space0.mesh
    reference_elements = mesh.elements._index_mapping._reference_elements

    if mesh.n == 2:
        xi, et = np.meshgrid(*quad_nodes, indexing='ij')
        xi = xi.ravel('F')
        et = et.ravel('F')
        detJ = mesh.elements.ct.Jacobian(xi, et)
    elif mesh.n == 3:
        xi, et, sg = np.meshgrid(*quad_nodes, indexing='ij')
        xi = xi.ravel('F')
        et = et.ravel('F')
        sg = sg.ravel('F')
        detJ = mesh.elements.ct.Jacobian(xi, et, sg)
    else:
        raise Exception()

    M_dict = dict()
    for re in reference_elements:
        m0 = rcM0[re]
        m1 = rcM1[re]

        dJi = detJ[re]
        metric = quad_weights * dJi

        if special_key is None:
            raise NotImplementedError()
        elif special_key == '(bf1, db)':
            m = _special_bf1_db(m0, m1, metric)
        else:
            raise NotImplementedError()

        M_dict[re] = csr_matrix(m)

    return M_dict


def _special_bf1_db(m0, m1, metric):
    """This case is for bundle 1-form inner diagonal-bundle-form, in 2D."""

    dim0 = _find_dim(m0)
    dim1 = _find_dim(m1)

    assert dim0 == [2, 2] and dim1 == [1], f"safety check"
    m00 = m0[0][0]
    m11 = m0[1][1]
    m1 = m1[0]
    o0 = np.einsum('ij, ik, i -> jk', m00, m1, metric, optimize='optimal')
    o1 = np.einsum('ij, ik, i -> jk', m11, m1, metric, optimize='optimal')
    return o0 + o1
