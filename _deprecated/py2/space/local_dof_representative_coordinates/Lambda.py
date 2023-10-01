# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen


class MseHyPy2LocalDofsRepresentativeCooLambda(Frozen):
    """Generation in-dependent."""

    def __init__(self, space):
        """Store required info."""
        self._space = space
        self._k = space.abstract.k
        self._ori = space.abstract.orientation
        self._freeze()

    def __call__(self, degree):
        """Find the coordinates for degree."""
        k = self._k
        if k == 1:
            return getattr(self, f"_k{k}_{self._ori}")(degree)
        else:
            return getattr(self, f"_k{k}")(degree)

    def _k0(self, degree):
        """"""
        nodes = self._space[degree].nodes
        xi, et = np.meshgrid(*nodes, indexing='ij')

        q_xi = xi.ravel('F')
        q_et = et.ravel('F')
        t_xi = np.concatenate(
            (
                np.array([-1, ]),
                xi[1:, :].ravel('F'),
            )
        )
        t_et = np.concatenate(
            (
                np.array([-1, ]),
                et[1:, :].ravel('F'),
            )
        )
        return {
            'q': [q_xi, q_et],
            't': [t_xi, t_et],
        }

    def _k1_inner(self, degree):
        """"""

        nodes = self._space[degree].nodes
        nodes_x, nodes_y = nodes

        dx_x = (nodes_x[1:] + nodes_x[:-1]) / 2
        dx_y = nodes_y
        dx_coo = np.meshgrid(dx_x, dx_y, indexing='ij')
        dx_x, dx_y = dx_coo
        dx_x_qt = dx_x.ravel('F')
        dx_y_qt = dx_y.ravel('F')

        dy_x = nodes_x
        dy_y = (nodes_y[1:] + nodes_y[:-1]) / 2
        dy_coo = np.meshgrid(dy_x, dy_y, indexing='ij')
        dy_x, dy_y = dy_coo
        dy_x_q = dy_x.ravel('F')
        dy_y_q = dy_y.ravel('F')
        dy_x_t = dy_x[1:, :].ravel('F')
        dy_y_t = dy_y[1:, :].ravel('F')

        return {
            'q': [
                np.concatenate([dx_x_qt, dy_x_q]),
                np.concatenate([dx_y_qt, dy_y_q]),
            ],
            't': [
                np.concatenate([dx_x_qt, dy_x_t]),
                np.concatenate([dx_y_qt, dy_y_t]),
            ]
        }

    def _k1_outer(self, degree):
        """"""
        nodes = self._space[degree].nodes
        nodes_x, nodes_y = nodes

        dx_x = (nodes_x[1:] + nodes_x[:-1]) / 2
        dx_y = nodes_y
        dx_coo = np.meshgrid(dx_x, dx_y, indexing='ij')
        dx_x, dx_y = dx_coo
        dx_x_qt = dx_x.ravel('F')
        dx_y_qt = dx_y.ravel('F')

        dy_x = nodes_x
        dy_y = (nodes_y[1:] + nodes_y[:-1]) / 2
        dy_coo = np.meshgrid(dy_x, dy_y, indexing='ij')
        dy_x, dy_y = dy_coo
        dy_x_q = dy_x.ravel('F')
        dy_y_q = dy_y.ravel('F')
        dy_x_t = dy_x[1:, :].ravel('F')
        dy_y_t = dy_y[1:, :].ravel('F')

        return {
            'q': [
                np.concatenate([dy_x_q, dx_x_qt]),
                np.concatenate([dy_y_q, dx_y_qt]),
            ],
            't': [
                np.concatenate([dy_x_t, dx_x_qt]),
                np.concatenate([dy_y_t, dx_y_qt]),
            ]
        }

    def _k2(self, degree):
        """"""
        nodes = self._space[degree].nodes
        nodes_x, nodes_y = nodes
        nodes_x = (nodes_x[:-1] + nodes_x[1:]) / 2
        nodes_y = (nodes_y[:-1] + nodes_y[1:]) / 2
        xi, et = np.meshgrid(nodes_x, nodes_y, indexing='ij')

        qt_xi = xi.ravel('F')
        qt_et = et.ravel('F')
        return {
            'q': [qt_xi, qt_et],
            't': [qt_xi, qt_et],
        }
