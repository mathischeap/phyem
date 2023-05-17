# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
"""
import sys
if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
from msepy.tools.gathering_matrix import RegularGatheringMatrix
import numpy as np


class MsePyGatheringMatrixLambda(Frozen):
    """"""

    def __init__(self, space):
        """Store required info."""
        self._space = space
        self._mesh = space.mesh
        self._k = space.abstract.k
        self._n = space.abstract.n  # manifold dimensions
        self._orientation = space.abstract.orientation
        self._cache = dict()
        self._freeze()

    def __call__(self, degree):
        """Making the local numbering for degree."""
        p = self._space[degree].p
        cache_key = str(p)  # only need p
        if cache_key in self._cache:
            gm = self._cache[cache_key]
        else:
            if self._n == 2 and self._k == 1:
                method_name = f"_n{self._n}_k{self._k}_{self._orientation}"
            else:
                method_name = f"_n{self._n}_k{self._k}"
            gm = getattr(self, method_name)(p)
            self._cache[cache_key] = gm

        return gm

    def _n3_k3(self, p):
        """"""
        total_num_elements = self._mesh.elements._num
        num_local_dofs = self._space.num_local_dofs.Lambda._n3_k3(p)
        total_num_dofs = total_num_elements * num_local_dofs
        gm = np.arange(0, total_num_dofs).reshape(
            (self._mesh.elements._num, num_local_dofs), order='C',
        )
        return RegularGatheringMatrix(gm)

    def _n3_k2(self, p):
        """A very old scheme, ugly but works."""
        gn0 = np.zeros((self._mesh.elements._num, p[0] + 1, p[1], p[2]), dtype=int)
        gn1 = np.zeros((self._mesh.elements._num, p[0], p[1] + 1, p[2]), dtype=int)
        gn2 = np.zeros((self._mesh.elements._num, p[0], p[1], p[2] + 1), dtype=int)
        element_map = self._mesh.elements.map
        current_number = 0
        num_basis_on_sides_dict = {'N': p[1]*p[2], 'S': p[1]*p[2],
                                   'W': p[0]*p[2], 'E': p[0]*p[2],
                                   'B': p[0]*p[1], 'F': p[0]*p[1]}
        for n in range(self._mesh.elements._num):  # currently, we are numbering nth element.
            # we first see the north side__________________________________________
            what_north = element_map[n][0]
            if isinstance(what_north, str):
                # the north side of nth element is on domain boundary. So we just number it.
                gn0[n, 0, :, :] = np.arange(current_number,
                                            current_number + num_basis_on_sides_dict['N']).reshape(
                    (p[1], p[2]), order='F')
                current_number += num_basis_on_sides_dict['N']
            elif what_north < n:
                # there is another element at nth element's north, and it is already numbered.
                # So we just take the numbering from it.
                gn0[n, 0, :, :] = gn0[what_north, -1, :, :]
            elif what_north > n:
                # there is another element at nth element's north
                # but, it is not numbered, so we number nth element's north
                gn0[n, 0, :, :] = np.arange(current_number,
                                            current_number + num_basis_on_sides_dict['N']).reshape(
                    (p[1], p[2]), order='F')
                current_number += num_basis_on_sides_dict['N']
            else:
                raise Exception()
            # now we number the internal dy^dz faces----------------------------
            gn0[n, 1:-1, :, :] = np.arange(current_number,
                                           current_number + (p[0]-1) * p[1] * p[2]).reshape(
                ((p[0]-1), p[1], p[2]), order='F')
            current_number += (p[0]-1)*p[1]*p[2]
            # next, we look at south side__________________________________________
            what_south = element_map[n][1]
            if isinstance(what_south, str):
                # the south side of nth element is on domain boundary. So we just number it.
                gn0[n, -1, :, :] = np.arange(current_number,
                                             current_number + num_basis_on_sides_dict['S']).reshape(
                    (p[1], p[2]), order='F')
                current_number += num_basis_on_sides_dict['S']
            elif what_south < n:
                # there is another element at nth element's south, and it is already numbered.
                # So we just take the numbering from it.
                gn0[n, -1, :, :] = gn0[what_south, 0, :, :]
            elif what_south > n:
                # there is another element at nth element's south
                # but, it is not numbered, so we number nth element's south
                gn0[n, -1, :, :] = np.arange(current_number,
                                             current_number + num_basis_on_sides_dict['S']).reshape(
                    (p[1], p[2]), order='F')
                current_number += num_basis_on_sides_dict['S']
            else:
                raise Exception()

            # we then look at the west side________________________________________
            what_west = element_map[n][2]
            if isinstance(what_west, str):
                # the west side of nth element is on domain boundary. So we just number it.
                gn1[n, :, 0, :] = np.arange(current_number,
                                            current_number + num_basis_on_sides_dict['W']).reshape(
                    (p[0], p[2]), order='F')
                current_number += num_basis_on_sides_dict['W']
            elif what_west < n:
                # there is another element at nth element's west, and it is already numbered.
                # So we just take the numbering from it.
                gn1[n, :, 0, :] = gn1[what_west, :, -1, :]
            elif what_west > n:
                # there is another element at nth element's west,
                # but, it is not numbered, so we number nth element's west.
                gn1[n, :, 0, :] = np.arange(current_number,
                                            current_number + num_basis_on_sides_dict['W']).reshape(
                    (p[0], p[2]), order='F')
                current_number += num_basis_on_sides_dict['W']
            else:
                raise Exception()
            # now we number the internal dz^dx faces----------------------------
            gn1[n, :, 1:-1, :] = np.arange(current_number,
                                           current_number + p[0] * (p[1]-1) * p[2]).reshape(
                (p[0], p[1]-1, p[2]), order='F')
            current_number += p[0]*(p[1]-1)*p[2]
            # next, we look at east side___________________________________________
            what_east = element_map[n][3]
            if isinstance(what_east, str):
                # the east side of nth element is on domain boundary. So we just number it.
                gn1[n, :, -1, :] = np.arange(current_number,
                                             current_number + num_basis_on_sides_dict['E']).reshape(
                    (p[0], p[2]), order='F')
                current_number += num_basis_on_sides_dict['E']
            elif what_east < n:
                # there is another element at nth element's east, and it is already numbered.
                # So we just take the numbering from it.
                gn1[n, :, -1, :] = gn1[what_east, :, 0, :]
            elif what_east > n:
                # there is another element at nth element's east
                # but, it is not numbered, so we number nth element's east
                gn1[n, :, -1, :] = np.arange(current_number,
                                             current_number + num_basis_on_sides_dict['E']).reshape(
                    (p[0], p[2]), order='F')
                current_number += num_basis_on_sides_dict['E']
            else:
                raise Exception()
            # we then look at the back side________________________________________
            what_back = element_map[n][4]
            if isinstance(what_back, str):
                # the back side of nth element is on domain boundary. So we just number it.
                gn2[n, :, :, 0] = np.arange(current_number,
                                            current_number + num_basis_on_sides_dict['B']).reshape(
                    (p[0], p[1]), order='F')
                current_number += num_basis_on_sides_dict['B']
            elif what_back < n:
                # there is another element at nth element's back and it is already numbered.
                # So we just take the numbering from it.
                gn2[n, :, :, 0] = gn2[what_back, :, :, -1]
            elif what_back > n:
                # there is another element at nth element's back,
                # but, it is not numbered, so we number nth element's back.
                gn2[n, :, :, 0] = np.arange(current_number,
                                            current_number + num_basis_on_sides_dict['B']).reshape(
                    (p[0], p[1]), order='F')
                current_number += num_basis_on_sides_dict['B']
            else:
                raise Exception()
            # now we number the internal dx^dy faces----------------------------
            gn2[n, :, :, 1:-1] = np.arange(current_number,
                                           current_number + p[0] * p[1] * (p[2]-1)).reshape(
                (p[0], p[1], p[2]-1), order='F')
            current_number += p[0]*p[1]*(p[2]-1)

            # next, we look at front side__________________________________________
            what_front = element_map[n][5]
            if isinstance(what_front, str):
                # the front side of nth element is on domain boundary. So we just number it.
                gn2[n, :, :, -1] = np.arange(current_number,
                                             current_number + num_basis_on_sides_dict['F']).reshape(
                    (p[0], p[1]), order='F')
                current_number += num_basis_on_sides_dict['F']
            elif what_front < n:
                # there is another element at nth element's front, and it is already numbered.
                # So we just take the numbering from it.
                gn2[n, :, :, -1] = gn2[what_front, :, :, 0]
            elif what_front > n:
                # there is another element at nth element's front
                # but, it is not numbered, so we number nth element's front
                gn2[n, :, :, -1] = np.arange(current_number,
                                             current_number + num_basis_on_sides_dict['F']).reshape(
                    (p[0], p[1]), order='F')
                current_number += num_basis_on_sides_dict['F']
            else:
                raise Exception()
        # --------- Numbering for hybrid or non-hybrid 2-forms is done.----------------
        gn = (gn0, gn1, gn2)
        gn = np.array([
            np.concatenate([gn[j][i, ...].ravel('F') for j in range(3)]) for i in range(self._mesh.elements._num)
        ])
        return RegularGatheringMatrix(gn)

    def _n3_k1(self, p):
        """A very old scheme, ugly but works."""
        gn0 = -np.ones((self._mesh.elements._num, p[0], p[1] + 1, p[2] + 1), dtype=int)
        gn1 = -np.ones((self._mesh.elements._num, p[0] + 1, p[1], p[2] + 1), dtype=int)
        gn2 = -np.ones((self._mesh.elements._num, p[0] + 1, p[1] + 1, p[2]), dtype=int)
        mesh = self._mesh
        GE = mesh.topology.edge_numbering
        GS = mesh.topology.side_numbering
        AE = mesh.topology.edge_attachment
        AS = mesh.topology.side_attachment
        px, py, pz = p
        AE_dx_numbered = []  # numbered edges for dx dofs
        AE_dy_numbered = []
        AE_dz_numbered = []
        AS_dx_numbered = []  # numbered faces for dx
        AS_dy_numbered = []
        AS_dz_numbered = []
        # what we are going to do is numbering edge dofs element by element
        cn = 0  # current numbering
        dt1 = {'W': 0, 'E': -1, 'B': 0, 'F': -1, 'N': 0, 'S': -1}
        AE_dz_now = -1
        for m in range(mesh.elements._num):  # we will go through all elements.
            # _______dx_____________________________________________________________
            # - WB edge -
            AE_dx_now = GE[m][0]
            if AE_dx_now not in AE_dx_numbered:
                AE_dx_numbered.append(AE_dx_now)
                element_edges = AE[AE_dx_now]
                for i_edge in element_edges:
                    ith, edge1, edge2 = self._m3n3k1_helper(i_edge)
                    assert edge1 in ('W', 'E') and edge2 in ('B', 'F')
                    gn0[ith, :, dt1[edge1], dt1[edge2]] = np.arange(cn, cn + px)
                cn += px
            else:
                pass
            # - B face -
            AS_dx_now = GS[m][4]
            if AS_dx_now not in AS_dx_numbered:
                AS_dx_numbered.append(AS_dx_now)
                element_sides = AS[AS_dx_now]
                for i_side in element_sides:
                    ith, side = i_side.split('-')
                    ith = int(ith)
                    assert side in ('B', 'F')
                    gn0[ith, :, 1:-1, dt1[side]] = np.arange(cn, cn + px * (py - 1)).reshape(
                        (px, py-1), order='F')
                cn += px*(py-1)
            else:
                pass
            # - EB edge -
            AE_dx_now = GE[m][1]
            if AE_dx_now not in AE_dx_numbered:
                AE_dx_numbered.append(AE_dx_now)
                element_edges = AE[AE_dx_now]
                for i_edge in element_edges:
                    ith, edge1, edge2 = self._m3n3k1_helper(i_edge)
                    assert edge1 in ('W', 'E') and edge2 in ('B', 'F')
                    gn0[ith, :, dt1[edge1], dt1[edge2]] = np.arange(cn, cn + px)
                cn += px
            else:
                pass
            # - W face -
            AS_dx_now = GS[m][2]
            if AS_dx_now not in AS_dx_numbered:
                AS_dx_numbered.append(AS_dx_now)
                element_sides = AS[AS_dx_now]
                for i_side in element_sides:
                    ith, side = i_side.split('-')
                    ith = int(ith)
                    assert side in ('W', 'E')
                    gn0[ith, :, dt1[side], 1:-1] = np.arange(cn, cn + px * (pz - 1)).reshape(
                        (px, pz-1), order='F')
                cn += px*(pz-1)
            else:
                pass
            # - internal -
            gn0[m, :, 1:-1, 1:-1] = np.arange(cn, cn + px * (py - 1) * (pz - 1)).reshape(
                (px, py-1, pz-1), order='F')
            cn += px*(py-1)*(pz-1)
            # - E face -
            AS_dx_now = GS[m][3]
            if AS_dx_now not in AS_dx_numbered:
                AS_dx_numbered.append(AS_dx_now)
                element_sides = AS[AS_dx_now]
                for i_side in element_sides:
                    ith, side = i_side.split('-')
                    ith = int(ith)
                    assert side in ('W', 'E')
                    gn0[ith, :, dt1[side], 1:-1] = np.arange(cn, cn + px * (pz - 1)).reshape(
                        (px, pz-1), order='F')
                cn += px*(pz-1)
            else:
                pass
            # - WF edge -
            AE_dx_now = GE[m][2]
            if AE_dx_now not in AE_dx_numbered:
                AE_dx_numbered.append(AE_dx_now)
                element_edges = AE[AE_dx_now]
                for i_edge in element_edges:
                    ith, edge1, edge2 = self._m3n3k1_helper(i_edge)
                    assert edge1 in ('W', 'E') and edge2 in ('B', 'F')
                    gn0[ith, :, dt1[edge1], dt1[edge2]] = np.arange(cn, cn + px)
                cn += px
            else:
                pass
            # - F face -
            AS_dx_now = GS[m][5]
            if AS_dx_now not in AS_dx_numbered:
                AS_dx_numbered.append(AS_dx_now)
                element_sides = AS[AS_dx_now]
                for i_side in element_sides:
                    ith, side = i_side.split('-')
                    ith = int(ith)
                    assert side in ('B', 'F')
                    gn0[ith, :, 1:-1, dt1[side]] = np.arange(cn, cn + px * (py - 1)).reshape(
                        (px, py-1), order='F')
                cn += px*(py-1)
            else:
                pass
            # - EF edge -
            AE_dx_now = GE[m][3]
            if AE_dx_now not in AE_dx_numbered:
                AE_dx_numbered.append(AE_dx_now)
                element_edges = AE[AE_dx_now]
                for i_edge in element_edges:
                    ith, edge1, edge2 = self._m3n3k1_helper(i_edge)
                    assert edge1 in ('W', 'E') and edge2 in ('B', 'F')
                    gn0[ith, :, dt1[edge1], dt1[edge2]] = np.arange(cn, cn + px)
                cn += px
            else:
                pass
            # _______dy_____________________________________________________________
            # - NB edge -
            AE_dy_now = GE[m][4]
            if AE_dy_now not in AE_dy_numbered:
                AE_dy_numbered.append(AE_dy_now)
                element_edges = AE[AE_dy_now]
                for i_edge in element_edges:
                    ith, edge1, edge2 = self._m3n3k1_helper(i_edge)
                    assert edge1 in ('N', 'S') and edge2 in ('B', 'F')
                    gn1[ith, dt1[edge1], :, dt1[edge2]] = np.arange(cn, cn + py)
                cn += py
            else:
                pass
            # - B face -
            AS_dy_now = GS[m][4]
            if AS_dy_now not in AS_dy_numbered:
                AS_dy_numbered.append(AS_dy_now)
                element_sides = AS[AS_dy_now]
                for i_side in element_sides:
                    ith, side = i_side.split('-')
                    ith = int(ith)
                    assert side in ('B', 'F')
                    gn1[ith, 1:-1, :, dt1[side]] = np.arange(cn, cn + (px - 1) * py).reshape(
                        (px-1, py), order='F')
                cn += (px-1)*py
            else:
                pass
            # - SB edge -
            AE_dy_now = GE[m][5]
            if AE_dy_now not in AE_dy_numbered:
                AE_dy_numbered.append(AE_dy_now)
                element_edges = AE[AE_dy_now]
                for i_edge in element_edges:
                    ith, edge1, edge2 = self._m3n3k1_helper(i_edge)
                    assert edge1 in ('N', 'S') and edge2 in ('B', 'F')
                    gn1[ith, dt1[edge1], :, dt1[edge2]] = np.arange(cn, cn + py)
                cn += py
            else:
                pass
            # - N face -
            AS_dy_now = GS[m][0]
            if AS_dy_now not in AS_dy_numbered:
                AS_dy_numbered.append(AS_dy_now)
                element_sides = AS[AS_dy_now]
                for i_side in element_sides:
                    ith, side = i_side.split('-')
                    ith = int(ith)
                    assert side in ('N', 'S')
                    gn1[ith, dt1[side], :, 1:-1] = np.arange(cn, cn + (pz - 1) * py).reshape(
                        (py, pz-1), order='F')
                cn += (pz-1)*py
            else:
                pass
            # - internal -
            gn1[m, 1:-1, :, 1:-1] = np.arange(cn, cn + (px - 1) * py * (pz - 1)).reshape(
                (px-1, py, pz-1), order='F')
            cn += (px-1)*py*(pz-1)
            # - S face -
            AS_dy_now = GS[m][1]
            if AS_dy_now not in AS_dy_numbered:
                AS_dy_numbered.append(AS_dy_now)
                element_sides = AS[AS_dy_now]
                for i_side in element_sides:
                    ith, side = i_side.split('-')
                    ith = int(ith)
                    assert side in ('N', 'S')
                    gn1[ith, dt1[side], :, 1:-1] = np.arange(cn, cn + (pz - 1) * py).reshape(
                        (py, pz-1), order='F')
                cn += (pz-1)*py
            else:
                pass
            # - NF edge -
            AE_dy_now = GE[m][6]
            if AE_dy_now not in AE_dy_numbered:
                AE_dy_numbered.append(AE_dy_now)
                element_edges = AE[AE_dy_now]
                for i_edge in element_edges:
                    ith, edge1, edge2 = self._m3n3k1_helper(i_edge)
                    assert edge1 in ('N', 'S') and edge2 in ('B', 'F')
                    gn1[ith, dt1[edge1], :, dt1[edge2]] = np.arange(cn, cn + py)
                cn += py
            else:
                pass
            # - F face -
            AS_dy_now = GS[m][5]
            if AS_dy_now not in AS_dy_numbered:
                AS_dy_numbered.append(AS_dy_now)
                element_sides = AS[AS_dy_now]
                for i_side in element_sides:
                    ith, side = i_side.split('-')
                    ith = int(ith)
                    assert side in ('B', 'F')
                    gn1[ith, 1:-1, :, dt1[side]] = np.arange(cn, cn + (px - 1) * py).reshape(
                        (px-1, py), order='F')
                cn += (px-1)*py
            else:
                pass
            # - SB edge -
            AE_dy_now = GE[m][7]
            if AE_dy_now not in AE_dy_numbered:
                AE_dy_numbered.append(AE_dy_now)
                element_edges = AE[AE_dy_now]
                for i_edge in element_edges:
                    ith, edge1, edge2 = self._m3n3k1_helper(i_edge)
                    assert edge1 in ('N', 'S') and edge2 in ('B', 'F')
                    gn1[ith, dt1[edge1], :, dt1[edge2]] = np.arange(cn, cn + py)
                cn += py
            else:
                pass
            # _______dz_____________________________________________________________
            # - NW edge -
            AE_dz_now = GE[m][8]
            if AE_dz_now not in AE_dz_numbered:
                AE_dz_numbered.append(AE_dz_now)
                element_edges = AE[AE_dz_now]
                for i_edge in element_edges:
                    ith, edge1, edge2 = self._m3n3k1_helper(i_edge)
                    assert edge1 in ('N', 'S') and edge2 in ('W', 'E')
                    gn2[ith, dt1[edge1], dt1[edge2], :] = np.arange(cn, cn + pz)
                cn += pz
            else:
                pass
            # - W face -
            AS_dz_now = GS[m][2]
            if AS_dz_now not in AS_dz_numbered:
                AS_dz_numbered.append(AS_dz_now)
                element_sides = AS[AS_dz_now]
                for i_side in element_sides:
                    ith, side = i_side.split('-')
                    ith = int(ith)
                    assert side in ('W', 'E')
                    gn2[ith, 1:-1, dt1[side], :] = np.arange(cn, cn + (px - 1) * pz).reshape(
                        (px-1, pz), order='F')
                cn += (px-1)*pz
            else:
                pass
            # - SW edge -
            AE_dz_now = GE[m][9]
            if AE_dz_now not in AE_dz_numbered:
                AE_dz_numbered.append(AE_dz_now)
                element_edges = AE[AE_dz_now]
                for i_edge in element_edges:
                    ith, edge1, edge2 = self._m3n3k1_helper(i_edge)
                    assert edge1 in ('N', 'S') and edge2 in ('W', 'E')
                    gn2[ith, dt1[edge1], dt1[edge2], :] = np.arange(cn, cn + pz)
                cn += pz
            else:
                pass
            # - N face -
            AS_dz_now = GS[m][0]
            if AS_dz_now not in AS_dz_numbered:
                AS_dz_numbered.append(AS_dz_now)
                element_sides = AS[AS_dz_now]
                for i_side in element_sides:
                    ith, side = i_side.split('-')
                    ith = int(ith)
                    assert side in ('N', 'S')
                    gn2[ith, dt1[side], 1:-1, :] = np.arange(cn, cn + (py - 1) * pz).reshape(
                        (py-1, pz), order='F')
                cn += (py-1)*pz
            else:
                pass
            # - internal -
            gn2[m, 1:-1, 1:-1, :] = np.arange(cn, cn + (px - 1) * (py - 1) * pz).reshape(
                (px-1, py-1, pz), order='F')
            cn += (px-1)*(py-1)*pz
            # - S face -
            AS_dz_now = GS[m][1]
            if AS_dz_now not in AS_dz_numbered:
                AS_dz_numbered.append(AS_dz_now)
                element_sides = AS[AS_dz_now]
                for i_side in element_sides:
                    ith, side = i_side.split('-')
                    ith = int(ith)
                    assert side in ('N', 'S')
                    gn2[ith, dt1[side], 1:-1, :] = np.arange(cn, cn + (py - 1) * pz).reshape(
                        (py-1, pz), order='F')
                cn += (py-1)*pz
            else:
                pass
            # - NE edge -
            AE_dz_now = GE[m][10]
            if AE_dz_now not in AE_dz_numbered:
                AE_dz_numbered.append(AE_dz_now)
                element_edges = AE[AE_dz_now]
                for i_edge in element_edges:
                    ith, edge1, edge2 = self._m3n3k1_helper(i_edge)
                    assert edge1 in ('N', 'S') and edge2 in ('W', 'E')
                    gn2[ith, dt1[edge1], dt1[edge2], :] = np.arange(cn, cn + pz)
                cn += pz
            else:
                pass
            # - W face -
            AS_dz_now = GS[m][3]
            if AS_dz_now not in AS_dz_numbered:
                AS_dz_numbered.append(AS_dz_now)
                element_sides = AS[AS_dz_now]
                for i_side in element_sides:
                    ith, side = i_side.split('-')
                    ith = int(ith)
                    assert side in ('W', 'E')
                    gn2[ith, 1:-1, dt1[side], :] = np.arange(cn, cn + (px - 1) * pz).reshape(
                        (px-1, pz), order='F')
                cn += (px-1)*pz
            else:
                pass
            # - SE edge -
            AE_dz_now = GE[m][11]
            if AE_dz_now not in AE_dz_numbered:
                AE_dz_numbered.append(AE_dz_now)
                element_edges = AE[AE_dz_now]
                for i_edge in element_edges:
                    ith, edge1, edge2 = self._m3n3k1_helper(i_edge)
                    assert edge1 in ('N', 'S') and edge2 in ('W', 'E')
                    gn2[ith, dt1[edge1], dt1[edge2], :] = np.arange(cn, cn + pz)
                cn += pz
            else:
                pass
            # ----------------------------------------------------------------------
        assert AE_dz_now == np.max(GE)
        # Now, numbering for non-hybrid 1-form is done.
        # We group what we have got.----------------------------------------------------
        gn = (gn0, gn1, gn2)
        gn = np.array([
            np.concatenate([gn[j][i, ...].ravel('F') for j in range(3)]) for i in range(self._mesh.elements._num)
        ])
        return RegularGatheringMatrix(gn)

    @staticmethod
    def _m3n3k1_helper(edge_indicator):
        """
        We use this method to find the location indices for an element edge
        labeled `edge_indicator`.

        Parameters
        ----------
        edge_indicator : str
            Like '0-WB', '0-EB', '10-SW' and so on.

        Returns
        -------
        ith : int
            The edge is in the `ith`th element.
        edge1 :
        edge2 :

        """
        ith, edges = edge_indicator.split('-')
        ith = int(ith)
        edge1, edge2 = edges
        return ith, edge1, edge2

    def _n3_k0(self, p):
        """A very old scheme, ugly but works."""
        mesh = self._mesh
        corner_position = mesh.topology.corner_attachment
        edge_position = mesh.topology.edge_attachment
        side_position = mesh.topology.side_attachment
        corner_gn = mesh.topology.corner_numbering
        edge_gn = mesh.topology.edge_numbering
        side_gn = mesh.topology.side_numbering
        gn = - np.ones((mesh.elements._num, p[0]+1, p[1]+1, p[2]+1), dtype=int)
        current_num = 0
        corner_index_dict = {
            'NWB': 0, 'SWB': 1, 'NEB': 2, 'SEB': 3, 'NWF': 4, 'SWF': 5, "NEF": 6, 'SEF': 7}
        edge_index_dict = {
            'WB': 0, 'EB': 1, 'WF': 2, 'EF': 3, 'NB': 4, 'SB': 5,
            'NF': 6, 'SF': 7, 'NW': 8, 'SW': 9, 'NE': 10, 'SE': 11}
        side_index_dict = {'N': 0, 'S': 1, 'W': 2, 'E': 3, 'B': 4, 'F': 5}
        for k in range(mesh.elements._num):
            # we go through all positions of each element.
            for position in ('NWB', 'WB', 'SWB', 'NB', 'B', 'SB', 'NEB', 'EB',
                             'SEB', 'NW', 'W', 'SW', 'NWF', 'WF', 'SWF', 'N', 'I',
                             'S', 'NE', 'E', 'SE', 'NF', 'F', 'SF',
                             'NEF', 'EF', 'SEF'):
                if position in ('NWB', 'SWB', 'NEB', 'SEB', 'NWF', 'SWF', 'NEF', 'SEF'):
                    triple_trace_element_no = corner_gn[k, corner_index_dict[position]]
                    triple_trace_element_position = corner_position[triple_trace_element_no]
                    gn, current_num = self.___number_triple_trace_element_position___(
                        gn, current_num, triple_trace_element_position
                    )
                elif position in ('WB', 'NB', 'SB', 'EB',
                                  'NW', 'SW', 'WF', 'NE',
                                  'SE', 'NF', 'SF', 'EF'):
                    dump_element_no = edge_gn[k, edge_index_dict[position]]
                    dump_element_position = edge_position[dump_element_no]
                    gn, current_num = self.___number_dump_element_position___(
                        p, gn, current_num, dump_element_position
                    )
                elif position in ('N', 'S', 'W', 'E', 'B', 'F'):
                    trace_element_no = side_gn[k, side_index_dict[position]]
                    trace_element_position = side_position[trace_element_no]
                    gn, current_num = self.___number_trace_element_position___(
                        p, gn, current_num, trace_element_position
                    )
                elif position == 'I':
                    PPP = (p[0]-1) * (p[1]-1) * (p[2]-1)
                    if PPP > 0:
                        gn[k, 1:-1, 1:-1, 1:-1] = np.arange(
                            current_num, current_num+PPP).reshape(
                            (p[0]-1, p[1]-1, p[2]-1), order='F')
                        current_num += PPP
                else:
                    raise Exception()

        gn = np.array([gn[j].ravel('F') for j in range(self._mesh.elements._num)])
        return RegularGatheringMatrix(gn)

    @staticmethod
    def ___number_triple_trace_element_position___(gn, current_num, triple_trace_element_position):
        """"""
        numbered = None
        for position_tt in triple_trace_element_position:
            if position_tt[0] == '<':
                pass
            else:
                mesh_element_no, position = position_tt.split('-')
                mesh_element_no = int(mesh_element_no)
                if position == 'NWB':
                    if gn[mesh_element_no, 0, 0, 0] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 0, 0, 0] = current_num
                elif position == 'SWB':
                    if gn[mesh_element_no, -1, 0, 0] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, -1, 0, 0] = current_num
                elif position == 'NEB':
                    if gn[mesh_element_no, 0, -1, 0] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 0, -1, 0] = current_num
                elif position == 'SEB':
                    if gn[mesh_element_no, -1, -1, 0] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, -1, -1, 0] = current_num
                elif position == 'NWF':
                    if gn[mesh_element_no, 0, 0, -1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 0, 0, -1] = current_num
                elif position == 'SWF':
                    if gn[mesh_element_no, -1, 0, -1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, -1, 0, -1] = current_num
                elif position == 'NEF':
                    if gn[mesh_element_no, 0, -1, -1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 0, -1, -1] = current_num
                elif position == 'SEF':
                    if gn[mesh_element_no, -1, -1, -1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, -1, -1, -1] = current_num
                else:
                    raise Exception()
        if numbered is None:
            current_num += 1
        return gn, current_num

    @staticmethod
    def ___number_dump_element_position___(p, gn, current_num, dump_element_position):
        """ """
        pxyz = p
        numbered = None
        p = None
        for position_d in dump_element_position:
            mesh_element_no, position = position_d.split('-')
            mesh_element_no = int(mesh_element_no)
            # ___ dz edges _________________________________________________________
            if position == 'NW':
                if p is None:
                    p = pxyz[2]
                else:
                    assert p == pxyz[2]
                if p == 1:
                    pass
                else:
                    if gn[mesh_element_no, 0, 0, 1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 0, 0, 1:-1] = np.arange(
                            current_num, current_num+p-1)
            elif position == 'SW':
                if p is None:
                    p = pxyz[2]
                else:
                    assert p == pxyz[2]
                if p == 1:
                    pass
                else:
                    if gn[mesh_element_no, -1, 0, 1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, -1, 0, 1:-1] = np.arange(
                            current_num, current_num+p-1)
            elif position == 'SE':
                if p is None:
                    p = pxyz[2]
                else:
                    assert p == pxyz[2]
                if p == 1:
                    pass
                else:
                    if gn[mesh_element_no, -1, -1, 1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, -1, -1, 1:-1] = np.arange(
                            current_num, current_num+p-1)
            elif position == 'NE':
                if p is None:
                    p = pxyz[2]
                else:
                    assert p == pxyz[2]
                if p == 1:
                    pass
                else:
                    if gn[mesh_element_no, 0, -1, 1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 0, -1, 1:-1] = np.arange(
                            current_num, current_num+p-1)
            # ___ dy edges _________________________________________________________
            elif position == 'NB':
                if p is None:
                    p = pxyz[1]
                else:
                    assert p == pxyz[1]
                if p == 1:
                    pass
                else:
                    if gn[mesh_element_no, 0, 1, 0] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 0, 1:-1, 0] = np.arange(
                            current_num, current_num+p-1)
            elif position == 'SB':
                if p is None:
                    p = pxyz[1]
                else:
                    assert p == pxyz[1]
                if p == 1:
                    pass
                else:
                    if gn[mesh_element_no, -1, 1, 0] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, -1, 1:-1, 0] = np.arange(
                            current_num, current_num+p-1)
            elif position == 'SF':
                if p is None:
                    p = pxyz[1]
                else:
                    assert p == pxyz[1]
                if p == 1:
                    pass
                else:
                    if gn[mesh_element_no, -1, 1, -1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, -1, 1:-1, -1] = np.arange(
                            current_num, current_num+p-1)
            elif position == 'NF':
                if p is None:
                    p = pxyz[1]
                else:
                    assert p == pxyz[1]
                if p == 1:
                    pass
                else:
                    if gn[mesh_element_no, 0, 1, -1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 0, 1:-1, -1] = np.arange(
                            current_num, current_num+p-1)
            # ___ dx edges _________________________________________________________
            elif position == 'WB':
                if p is None:
                    p = pxyz[0]
                else:
                    assert p == pxyz[0]
                if p == 1:
                    pass
                else:
                    if gn[mesh_element_no, 1, 0, 0] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 1:-1, 0, 0] = np.arange(
                            current_num, current_num+p-1)
            elif position == 'EB':
                if p is None:
                    p = pxyz[0]
                else:
                    assert p == pxyz[0]
                if p == 1:
                    pass
                else:
                    if gn[mesh_element_no, 1, -1, 0] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 1:-1, -1, 0] = np.arange(
                            current_num, current_num+p-1)
            elif position == 'WF':
                if p is None:
                    p = pxyz[0]
                else:
                    assert p == pxyz[0]
                if p == 1:
                    pass
                else:
                    if gn[mesh_element_no, 1, 0, -1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 1:-1, 0, -1] = np.arange(
                            current_num, current_num+p-1)
            elif position == 'EF':
                if p is None:
                    p = pxyz[0]
                else:
                    assert p == pxyz[0]
                if p == 1:
                    pass
                else:
                    if gn[mesh_element_no, 1, -1, -1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 1:-1, -1, -1] = np.arange(
                            current_num, current_num+p-1)
            # __ ELSE: ERRORING ____________________________________________________
            else:
                raise Exception()
            # ----------------------------------------------------------------------
        if numbered is None:
            current_num += p - 1
        return gn, current_num

    @staticmethod
    def ___number_trace_element_position___(p, gn, current_num, trace_element_position):
        """ """
        pxyz = p
        numbered = None
        p, p1, p2 = None, None, None
        for position_t in trace_element_position:
            mesh_element_no, position = position_t.split('-')
            mesh_element_no = int(mesh_element_no)
            if position == 'N':
                if p is None:
                    p1, p2 = (pxyz[1]-1), (pxyz[2]-1)
                    p = p1 * p2
                else:
                    assert p1 == (pxyz[1]-1) and p2 == (pxyz[2]-1)
                if p == 0:
                    pass
                else:
                    if gn[mesh_element_no, 0, 1, 1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 0, 1:-1, 1:-1] = np.arange(
                            current_num, current_num+p).reshape(
                            (p1, p2), order='F')
            elif position == 'S':
                if p is None:
                    p1, p2 = (pxyz[1]-1), (pxyz[2]-1)
                    p = p1 * p2
                else:
                    assert p1 == (pxyz[1]-1) and p2 == (pxyz[2]-1)
                if p == 0:
                    pass
                else:
                    if gn[mesh_element_no, -1, 1, 1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, -1, 1:-1, 1:-1] = np.arange(
                            current_num, current_num+p).reshape(
                            (p1, p2), order='F')
            elif position == 'W':
                if p is None:
                    p1, p2 = (pxyz[0]-1), (pxyz[2]-1)
                    p = p1 * p2
                else:
                    assert p1 == (pxyz[0]-1) and p2 == (pxyz[2]-1)
                if p == 0:
                    pass
                else:
                    if gn[mesh_element_no, 1, 0, 1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 1:-1, 0, 1:-1] = np.arange(
                            current_num, current_num+p).reshape(
                            (p1, p2), order='F')
            elif position == 'E':
                if p is None:
                    p1, p2 = (pxyz[0]-1), (pxyz[2]-1)
                    p = p1 * p2
                else:
                    assert p1 == (pxyz[0]-1) and p2 == (pxyz[2]-1)
                if p == 0:
                    pass
                else:
                    if gn[mesh_element_no, 1, -1, 1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 1:-1, -1, 1:-1] = np.arange(
                            current_num, current_num+p).reshape(
                            (p1, p2), order='F')
            elif position == 'B':
                if p is None:
                    p1, p2 = (pxyz[0]-1), (pxyz[1]-1)
                    p = p1 * p2
                else:
                    assert p1 == (pxyz[0]-1) and p2 == (pxyz[1]-1)
                if p == 0:
                    pass
                else:
                    if gn[mesh_element_no, 1, 1, 0] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 1:-1, 1:-1, 0] = np.arange(
                            current_num, current_num+p).reshape(
                            (p1, p2), order='F')
            elif position == 'F':
                if p is None:
                    p1, p2 = (pxyz[0]-1), (pxyz[1]-1)
                    p = p1 * p2
                else:
                    assert p1 == (pxyz[0]-1) and p2 == (pxyz[1]-1)
                if p == 0:
                    pass
                else:
                    if gn[mesh_element_no, 1, 1, -1] != -1:
                        if numbered is None:
                            numbered = True
                        else:
                            assert numbered
                    else:
                        gn[mesh_element_no, 1:-1, 1:-1, -1] = np.arange(
                            current_num, current_num+p).reshape(
                            (p1, p2), order='F')
            else:
                raise Exception()
        if numbered is None:
            current_num += p
        return gn, current_num

    def _n2_k0(self, p):
        """A very old scheme, ugly but works."""
        # the idea of numbering 0-form on 2-manifold is the following
        # 1) we go through all elements
        # 2) we check its UL corner and number the dof
        # 3) We check its L edge and number the dofs
        # 4) we check its DL corner and number the dof
        # 5) We check its U edge and number the dofs
        # 6) we number internal dofs
        # 7) We check its D edge and number the dofs
        # 8) we check its UR corner and number the dof
        # 9) we check its R edge and number the dofs
        # 10) we check its DR corner and number the dof

        mp = self._mesh.elements.map
        gm = - np.ones((self._mesh.elements._num, p[0] + 1, p[1] + 1), dtype=int)
        _dict_ = {'U': (0, 0), 'D': (0, -1), 'L': (1, 0), 'R': (1, -1)}
        _cd_ = {'UL': (0, 0), 'DL': (-1, 0), 'UR': (0, -1), 'DR': (-1, -1)}
        _n2id_ = {'UL': 0, 'DL': 1, 'UR': 2, 'DR': 3}
        edge_pair = {0: 1, 1: 0, 2: 3, 3: 2}
        ind = {'U': 0, 'D': 1, 'L': 2, 'R': 3}
        num_dof_dict = {'L': p[0] - 1, 'U': p[1] - 1, 'D': p[1] - 1, 'R': p[0] - 1}
        ngc = self._mesh.topology.corner_numbering
        gac = self._mesh.topology.corner_attachment

        current_num = 0
        for i in range(self._mesh.elements._num):  # do step 1).
            for number_where in ('UL', 'L', 'DL', 'U', 'I', 'D', 'UR', 'R', 'DR'):
                # ________ element corners ________________________________________
                if number_where in ('UL', 'DL', 'UR', 'DR'):  # change tuple to change sequence
                    index_x, index_y = _cd_[number_where]
                    if gm[i, index_x, index_y] != -1:  # this corner numbered
                        pass  # do nothing, as it is numbered
                    else:  # not numbered, we number it.
                        attachment = gac[ngc[i][_n2id_[number_where]]]
                        for numbering_element_numbering_corner in attachment:
                            numbering_element, numbering_corner = \
                                numbering_element_numbering_corner.split('-')
                            numbering_element = int(numbering_element)
                            index_x, index_y = _cd_[numbering_corner]
                            gm[numbering_element, index_x, index_y] = current_num
                        current_num += 1
                # _______ element edges (except corners) __________________________
                elif number_where in ('L', 'U', 'D', 'R'):  # change tuple to change sequence
                    numbering_element = i
                    numbering_edge_id = ind[number_where]
                    attached_2_numbering_edge = mp[numbering_element][numbering_edge_id]
                    # _____ element edge on domain boundary________________________
                    if attached_2_numbering_edge == -1:
                        # the numbering edge is on domain boundary
                        axis, start_end = _dict_[number_where]
                        if axis == 0:
                            gm[i, start_end, 1:-1] = np.arange(
                                current_num, current_num + num_dof_dict[number_where])
                        elif axis == 1:
                            gm[i, 1:-1, start_end] = np.arange(
                                current_num, current_num + num_dof_dict[number_where])
                        else:
                            raise Exception()
                        current_num += num_dof_dict[number_where]
                    # ___ element edge attached to another mesh element____________
                    else:
                        attached_element = attached_2_numbering_edge
                        attached_edge_id = edge_pair[numbering_edge_id]
                        assert edge_pair[attached_edge_id] == numbering_edge_id
                        # __ another mesh element is not numbered yet _____________
                        if attached_element > numbering_element:
                            # the attached_edge can not be numbered, we number numbering_edge
                            axis, start_end = _dict_[number_where]
                            if axis == 0:
                                gm[i, start_end, 1:-1] = np.arange(
                                    current_num, current_num + num_dof_dict[number_where])
                            elif axis == 1:
                                gm[i, 1:-1, start_end] = np.arange(
                                    current_num, current_num + num_dof_dict[number_where])
                            else:
                                raise Exception()
                            current_num += num_dof_dict[number_where]
                        # __another mesh element is numbered_______________________
                        else:  # we take the numbering from attached_edge
                            axis, start_end = _dict_[number_where]
                            attached_se = {0: -1, -1: 0}[start_end]
                            if axis == 0:
                                gm[i, start_end, 1:-1] = gm[attached_element][attached_se, 1:-1]
                            elif axis == 1:
                                gm[i, 1:-1, start_end] = gm[attached_element][1:-1, attached_se]
                            else:
                                raise Exception()
                # _____ internal corners __________________________________________
                elif number_where == 'I':
                    gm[i, 1:-1, 1:-1] = np.arange(
                        current_num, current_num+(p[0]-1)*(p[1]-1)).reshape((p[0]-1, p[1]-1), order='F')
                    current_num += (p[0]-1)*(p[1]-1)
                # ____ ELSE _____________________________________________
                else:
                    raise Exception(f"cannot reach here!")

        gm = np.array([gm[j].ravel('F') for j in range(self._mesh.elements._num)])
        return RegularGatheringMatrix(gm)

    def _n2_k1_inner(self, p):
        """A very old scheme, ugly but works.
        An old scheme. it is slow. But since we only do it once, we keep it.
        """
        # the idea of numbering non-hybrid outer-1form is the following:
        # 1) we go through all elements
        # 2) we check its L edge and number dx dofs
        # 3) we number internal dx dofs
        # 4) we check its R edge and number dx dofs
        # 5) we check its U edge and number dy dofs
        # 6) we number internal dy dofs
        # 7) we check its D edge and number dy dofs

        mp = self._mesh.elements.map
        ind = {'U': 0, 'D': 1, 'L': 2, 'R': 3}
        num_dof_dict = {'U': p[1], 'D': p[1], 'L': p[0], 'R': p[0]}
        _dict_ = {'U': (0, 0), 'D': (0, -1), 'L': (1, 0), 'R': (1, -1)}
        edge_pair = {0: 1, 1: 0, 2: 3, 3: 2}
        gm_dx = - np.ones((self._mesh.elements._num, p[0], p[1] + 1), dtype=int)
        gm_dy = - np.ones((self._mesh.elements._num, p[0] + 1, p[1]), dtype=int)
        current_num = 0
        for i in range(self._mesh.elements._num):  # do step 1).
            for number_where in ('L', 'dxI', 'R', 'U', 'dyI', 'D'):
                # ________ element edges __________________________________________
                if number_where in ('L', 'R', 'U', 'D'):
                    numbering_element = i
                    numbering_edge_id = ind[number_where]
                    attached_2_numbering_edge = mp[numbering_element][numbering_edge_id]
                    # _____ element edge on domain boundary________________________
                    if attached_2_numbering_edge == -1:
                        # the numbering edge is on domain boundary
                        axis, start_end = _dict_[number_where]
                        if axis == 0:
                            gm_dy[i][start_end, :] = np.arange(
                                current_num, current_num + num_dof_dict[number_where])
                        elif axis == 1:
                            gm_dx[i][:, start_end] = np.arange(
                                current_num, current_num + num_dof_dict[number_where])
                        else:
                            raise Exception()
                        current_num += num_dof_dict[number_where]
                    # ____ element edge attached to another mesh element____________
                    else:
                        attached_element = attached_2_numbering_edge
                        attached_edge_id = edge_pair[numbering_edge_id]
                        assert edge_pair[attached_edge_id] == numbering_edge_id
                        # ___ another mesh element is not numbered yet _____________
                        if attached_element > numbering_element:
                            # the attached_edge can not be numbered, we number numbering_edge
                            axis, start_end = _dict_[number_where]
                            if axis == 0:
                                gm_dy[i][start_end, :] = np.arange(
                                    current_num, current_num + num_dof_dict[number_where])
                            elif axis == 1:
                                gm_dx[i][:, start_end] = np.arange(
                                    current_num, current_num + num_dof_dict[number_where])
                            else:
                                raise Exception()
                            current_num += num_dof_dict[number_where]
                        # ___another mesh element is numbered_______________________
                        else:  # we take the numbering from attached_edge
                            axis, start_end = _dict_[number_where]
                            attached_se = {0: -1, -1: 0}[start_end]
                            if axis == 0:
                                gm_dy[i][start_end, :] = gm_dy[attached_element][attached_se, :]
                            elif axis == 1:
                                gm_dx[i][:, start_end] = gm_dx[attached_element][:, attached_se]
                            else:
                                raise Exception()
                # _____ internal dx edges _________________________________________
                elif number_where == 'dxI':
                    gm_dx[i][:, 1:-1] = np.arange(
                        current_num, current_num+(p[1]-1)*p[0]).reshape((p[0], p[1]-1), order='F')
                    current_num += (p[1]-1)*p[0]
                # _____ internal dy edges _________________________________________
                elif number_where == 'dyI':
                    gm_dy[i][1:-1, :] = np.arange(
                        current_num, current_num+(p[0]-1)*p[1]).reshape((p[0]-1, p[1]), order='F')
                    current_num += (p[0]-1)*p[1]
                # ____ ELSE: _____________________________________________
                else:
                    raise Exception()
        gm = (gm_dx, gm_dy)
        gm = np.array([
            np.concatenate([gm[j][i, ...].ravel('F') for j in range(2)]) for i in range(self._mesh.elements._num)
        ])
        return RegularGatheringMatrix(gm)

    def _n2_k1_outer(self, p):
        """A very old scheme, ugly but works.
        An old scheme. it is slow. But since we only do it once, we keep it.
        """
        # the idea of numbering non-hybrid outer-1form is the following:
        # 1) we go through all elements
        # 2) we check its U edge and number dy dofs
        # 3) we number internal dy dofs
        # 4) we check its D edge and number dy dofs
        # 5) we check its L edge and number dx dofs
        # 6) we number internal dx dofs
        # 7) we check its R edge and number dx dofs
        mp = self._mesh.elements.map
        ind = {'U': 0, 'D': 1, 'L': 2, 'R': 3}
        num_dof_dict = {'U': p[1], 'D': p[1], 'L': p[0], 'R': p[0]}
        _dict_ = {'U': (0, 0), 'D': (0, -1), 'L': (1, 0), 'R': (1, -1)}
        edge_pair = {0: 1, 1: 0, 2: 3, 3: 2}
        men = self._mesh.elements._num
        gm_dy = - np.ones((men, p[0] + 1, p[1]), dtype=int)
        gm_dx = - np.ones((men, p[0], p[1] + 1), dtype=int)
        current_num = 0
        for i in range(self._mesh.elements._num):  # do step 1).
            for number_where in ('U', 'dyI', 'D', 'L', 'dxI', 'R'):
                # _________ element edges __________________________________________
                if number_where in ('U', 'D', 'L', 'R'):
                    numbering_element = i
                    numbering_edge_id = ind[number_where]
                    attached_2_numbering_edge = mp[numbering_element][numbering_edge_id]
                    # ______ element edge on domain boundary________________________
                    if attached_2_numbering_edge == -1:
                        # the numbering edge is on domain boundary
                        axis, start_end = _dict_[number_where]
                        if axis == 0:
                            gm_dy[i][start_end, :] = np.arange(
                                current_num, current_num + num_dof_dict[number_where])
                        elif axis == 1:
                            gm_dx[i][:, start_end] = np.arange(
                                current_num, current_num + num_dof_dict[number_where])
                        else:
                            raise Exception()
                        current_num += num_dof_dict[number_where]
                    # ____ element edge attached to another mesh element____________
                    else:
                        attached_element = attached_2_numbering_edge
                        attached_edge_id = edge_pair[numbering_edge_id]
                        assert edge_pair[attached_edge_id] == numbering_edge_id
                        # ___ another mesh element is not numbered yet _____________
                        if attached_element > numbering_element:
                            # the attached_edge can not be numbered, we number numbering_edge
                            axis, start_end = _dict_[number_where]
                            if axis == 0:
                                gm_dy[i][start_end, :] = np.arange(
                                    current_num, current_num + num_dof_dict[number_where])
                            elif axis == 1:
                                gm_dx[i][:, start_end] = np.arange(
                                    current_num, current_num + num_dof_dict[number_where])
                            else:
                                raise Exception()
                            current_num += num_dof_dict[number_where]
                        # ___another mesh element is numbered_______________________
                        else:  # we take the numbering from attached_edge
                            axis, start_end = _dict_[number_where]
                            attached_se = {0: -1, -1: 0}[start_end]
                            if axis == 0:
                                gm_dy[i][start_end, :] = gm_dy[attached_element][attached_se, :]
                            elif axis == 1:
                                gm_dx[i][:, start_end] = gm_dx[attached_element][:, attached_se]
                            else:
                                raise Exception()
                # _____ internal dy edges _________________________________________
                elif number_where == 'dyI':
                    gm_dy[i][1:-1, :] = np.arange(
                        current_num, current_num+(p[0]-1)*p[1]).reshape((p[0]-1, p[1]), order='F')
                    current_num += (p[0]-1)*p[1]
                # _____ internal dx edges _________________________________________
                elif number_where == 'dxI':
                    gm_dx[i][:, 1:-1] = np.arange(
                        current_num, current_num+(p[1]-1)*p[0]).reshape((p[0], p[1]-1), order='F')
                    current_num += (p[1]-1)*p[0]
                # _____ ELSE ____________________________________________
                else:
                    raise Exception()
        gm = (gm_dy, gm_dx)
        gm = np.array([
            np.concatenate([gm[j][i, ...].ravel('F') for j in range(2)]) for i in range(self._mesh.elements._num)
        ])
        return RegularGatheringMatrix(gm)

    def _n2_k2(self, p):
        """"""
        total_num_elements = self._mesh.elements._num
        num_local_dofs = self._space.num_local_dofs.Lambda._n2_k2(p)
        total_num_dofs = total_num_elements * num_local_dofs
        gm = np.arange(0, total_num_dofs).reshape(
            (self._mesh.elements._num, num_local_dofs), order='C',
        )
        return RegularGatheringMatrix(gm)

    def _n1_k0(self, p):
        """"""
        element_map = self._mesh.elements.map
        gm = - np.ones((self._mesh.elements._num, self._space.num_local_dofs.Lambda._n1_k0(p)), dtype=int)
        current = 0
        p = p[0]
        for e, mp in enumerate(element_map):
            # number x- node
            x_m = mp[0]
            if x_m == -1 or x_m > e:  # x- side of element #e is a boundary or not numbered
                gm[e, 0] = current
                current += 1
            else:
                gm[e, 0] = gm[x_m, -1]
            # node intermediate nodes
            gm[e, 1:-1] = np.arange(current, current + p - 1)
            current += p - 1

            # number x+ node
            x_p = mp[-1]
            if x_p == -1 or x_p > e:
                gm[e, -1] = current
                current += 1
            else:
                gm[e, -1] = gm[x_p, 0]
        return RegularGatheringMatrix(gm)

    def _n1_k1(self, p):
        """"""
        total_num_elements = self._mesh.elements._num
        num_local_dofs = self._space.num_local_dofs.Lambda._n1_k1(p)
        total_num_dofs = total_num_elements * num_local_dofs
        gm = np.arange(0, total_num_dofs).reshape(
            (self._mesh.elements._num, num_local_dofs), order='C',
        )
        return RegularGatheringMatrix(gm)
