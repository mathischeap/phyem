# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
import numpy as np

from generic.py.gathering_matrix import PyGM


class GatheringMatrixLambda(Frozen):
    """"""

    def __init__(self, space):
        """Store required info."""
        self._space = space
        self._mesh = space.mesh
        self._k = space.abstract.k
        self._orientation = space.abstract.orientation
        self._cache = {}
        self._cache_000 = {}
        self._freeze()

    def __call__(self, degree):
        """Making the local numbering for degree."""
        p = self._space[degree].p
        if p in self._cache:
            return self._cache[p]
        else:
            if self._k == 1:
                method_name = f"_k{self._k}_{self._orientation}"
            else:
                method_name = f"_k{self._k}"
            gm = getattr(self, method_name)(p)
            self._cache[p] = gm
            return gm

    def _k0(self, p):
        """"""
        if p in self._cache_000:
            # another level of cache for 0-form as it is used for visualization particularly.
            return self._cache_000[p]
        else:
            pass

        element_map = self._mesh.map
        numbered_edges = dict()
        numbered_corner = dict()
        NUMBERING = dict()
        current = 0
        num_internal = (p-1) * (p-1)
        num_edge = p-1  # corners are numbered independently.

        for index in self._mesh:
            vertices = element_map[index]
            element = self._mesh[index]
            ele_type = element.type
            if ele_type == 'q':

                edge_dx_y0 = (vertices[0], vertices[1])
                edge_dx_y1 = (vertices[3], vertices[2])

                edge_dy_x0 = (vertices[0], vertices[3])
                edge_dy_x1 = (vertices[1], vertices[2])

                corner0 = vertices[0]
                corner1 = vertices[1]
                corner2 = vertices[2]
                corner3 = vertices[3]

            elif ele_type == 't':

                edge_dx_y0 = vertices[0], vertices[1]
                edge_dx_y1 = (vertices[0], vertices[2])

                edge_dy_x0 = None
                edge_dy_x1 = (vertices[1], vertices[2])

                corner0 = vertices[0]
                corner1 = vertices[1]
                corner2 = vertices[2]
                corner3 = None

            else:
                raise Exception()

            X = np.zeros([p+1, p+1], dtype=int)

            # vertex 0 / corner 0
            current, numbering = self._find_numering_corner(corner0, current, numbered_corner)
            X[0, 0] = numbering

            # edge 0, dx y0
            current, numbering = self._find_numering_along(edge_dx_y0, num_edge, current, numbered_edges)
            X[1:-1, 0] = numbering

            # vertex 1/ corner 1
            current, numbering = self._find_numering_corner(corner1, current, numbered_corner)
            X[-1, 0] = numbering

            # edge 3, dy x0
            current, numbering = self._find_numering_along(edge_dy_x0, num_edge, current, numbered_edges)
            if numbering is None:
                pass
            else:
                X[0, 1:-1] = numbering

            # internal
            numbering = np.arange(current, current+num_internal).reshape((p-1, p-1), order='F')
            current += num_internal
            X[1:-1, 1:-1] = numbering

            # edge 1, dy x1
            current, numbering = self._find_numering_along(edge_dy_x1, num_edge, current, numbered_edges)
            X[-1, 1:-1] = numbering

            # vertex 3 / corner 3
            current, numbering = self._find_numering_corner(corner3, current, numbered_corner)
            if numbering is None:
                pass
            else:
                X[0, -1] = numbering

            # edge2, dx y1
            current, numbering = self._find_numering_along(edge_dx_y1, num_edge, current, numbered_edges)
            X[1:-1, -1] = numbering

            # vertex 2 / corner 2
            current, numbering = self._find_numering_corner(corner2, current, numbered_corner)
            X[-1, -1] = numbering

            if ele_type == 'q':
                NUMBERING[index] = X.ravel('F')
            elif ele_type == 't':
                x0 = np.array([X[0, 0], ])
                NUMBERING[index] = np.concatenate(
                    [x0, X[1:, :].ravel('F')]
                )

            else:
                raise Exception()

        gm = PyGM(NUMBERING)
        self._cache_000[p] = gm
        return gm

    @staticmethod
    def _find_numering_corner(vertex, current, numbered_corner):
        """"""
        if vertex is None:
            numbering = None
        else:
            if vertex in numbered_corner:
                numbering = numbered_corner[vertex]
            else:
                numbering = current
                current += 1
                numbered_corner[vertex] = numbering

        return current, numbering

    def _k1_inner(self, p):
        """"""
        element_map = self._mesh.map
        numbered_edges = dict()
        NUMBERING = dict()
        current = 0
        num_internal = p * (p-1)
        num_edge = p

        for index in self._mesh:
            vertices = element_map[index]
            element = self._mesh[index]
            ele_type = element.type
            if ele_type == 'q':

                edge_dx_y0 = (vertices[0], vertices[1])
                edge_dx_y1 = (vertices[3], vertices[2])

                edge_dy_x0 = (vertices[0], vertices[3])
                edge_dy_x1 = (vertices[1], vertices[2])

            elif ele_type == 't':

                edge_dx_y0 = vertices[0], vertices[1]
                edge_dx_y1 = (vertices[0], vertices[2])

                edge_dy_x0 = None
                edge_dy_x1 = (vertices[1], vertices[2])

            else:
                raise Exception()

            DX = np.zeros([p, p+1], dtype=int)
            DY = np.zeros([p+1, p], dtype=int)

            current, numbering = self._find_numering_along(edge_dx_y0, num_edge, current, numbered_edges)
            DX[:, 0] = numbering

            numbering = np.arange(current, current+num_internal).reshape((p, p-1), order='F')
            current += num_internal
            DX[:, 1:-1] = numbering

            current, numbering = self._find_numering_along(edge_dx_y1, num_edge, current, numbered_edges)
            DX[:, -1] = numbering

            current, numbering = self._find_numering_along(edge_dy_x0, num_edge, current, numbered_edges)
            if numbering is None:
                pass
            else:
                DY[0, :] = numbering

            numbering = np.arange(current, current+num_internal).reshape((p-1, p), order='F')
            current += num_internal
            DY[1:-1, :] = numbering

            current, numbering = self._find_numering_along(edge_dy_x1, num_edge, current, numbered_edges)
            DY[-1, :] = numbering

            if ele_type == 'q':
                NUMBERING[index] = np.concatenate(
                    [DX.ravel('F'), DY.ravel('F')]
                )
            elif ele_type == 't':
                NUMBERING[index] = np.concatenate(
                    [DX.ravel('F'), DY[1:, :].ravel('F')]
                )
            else:
                raise Exception()

        return PyGM(NUMBERING)

    def _k1_outer(self, p):
        """"""
        element_map = self._mesh.map
        numbered_edges = dict()
        NUMBERING = dict()
        current = 0
        num_internal = p * (p-1)
        num_edge = p

        for index in self._mesh:
            vertices = element_map[index]
            element = self._mesh[index]
            ele_type = element.type
            if ele_type == 'q':

                edge_dy_x0 = (vertices[0], vertices[3])
                edge_dy_x1 = (vertices[1], vertices[2])

                edge_dx_y0 = (vertices[0], vertices[1])
                edge_dx_y1 = (vertices[3], vertices[2])

            elif ele_type == 't':
                edge_dy_x0 = None
                edge_dy_x1 = (vertices[1], vertices[2])

                edge_dx_y0 = vertices[0], vertices[1]
                edge_dx_y1 = (vertices[0], vertices[2])

            else:
                raise Exception()

            DY = np.zeros([p+1, p], dtype=int)
            DX = np.zeros([p, p+1], dtype=int)

            current, numbering = self._find_numering_along(edge_dy_x0, num_edge, current, numbered_edges)
            if numbering is None:
                pass
            else:
                DY[0, :] = numbering

            numbering = np.arange(current, current+num_internal).reshape((p-1, p), order='F')
            current += num_internal
            DY[1:-1, :] = numbering

            current, numbering = self._find_numering_along(edge_dy_x1, num_edge, current, numbered_edges)
            DY[-1, :] = numbering

            current, numbering = self._find_numering_along(edge_dx_y0, num_edge, current, numbered_edges)
            DX[:, 0] = numbering

            numbering = np.arange(current, current+num_internal).reshape((p, p-1), order='F')
            current += num_internal
            DX[:, 1:-1] = numbering

            current, numbering = self._find_numering_along(edge_dx_y1, num_edge, current, numbered_edges)
            DX[:, -1] = numbering

            if ele_type == 'q':
                NUMBERING[index] = np.concatenate(
                    [DY.ravel('F'), DX.ravel('F')]
                )
            elif ele_type == 't':
                NUMBERING[index] = np.concatenate(
                    [DY[1:, :].ravel('F'), DX.ravel('F')]
                )
            else:
                raise NotImplementedError()

        return PyGM(NUMBERING)

    @staticmethod
    def _find_numering_along(vertex_0_1, num_edge, current, numbered_edges):
        """"""
        if vertex_0_1 is None:
            numbering = None
        else:
            v0, v1 = vertex_0_1
            if (v0, v1) in numbered_edges:
                numbering = numbered_edges[(v0, v1)]
            elif (v1, v0) in numbered_edges:
                numbering = numbered_edges[(v1, v0)][::-1]
            else:
                numbering = [_ for _ in range(current, current+num_edge)]
                current += num_edge
                numbered_edges[(v0, v1)] = numbering

        return current, numbering

    def _k2(self, p):
        """"""
        NUMBERING = dict()
        current = 0
        local_dofs = p * p

        for index in self._mesh:
            NUMBERING[index] = np.arange(current, current+local_dofs)
            current += local_dofs

        return PyGM(NUMBERING)
