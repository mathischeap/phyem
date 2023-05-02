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
        """"""
        raise NotImplementedError

    def _n3_k1(self, p):
        """"""
        raise NotImplementedError

    def _n3_k0(self, p):
        """"""
        raise NotImplementedError

    def _n2_k0(self, p):
        """"""
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
        gac = self._mesh.topology.corners_attachment

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
                        # ---------------------------------------------------------
                    # -------------------------------------------------------------
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
        """An old scheme. it is slow. But since we only do it once, we keep it."""
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
                        # ---------------------------------------------------------
                    # -------------------------------------------------------------
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
                # ------------------------------------------------------------------
        # --------------------------------------------------------------------------
        gm = (gm_dx, gm_dy)
        gm = np.array([
            np.concatenate([gm[j][i, ...].ravel('F') for j in range(2)]) for i in range(self._mesh.elements._num)
        ])
        return RegularGatheringMatrix(gm)

    def _n2_k1_outer(self, p):
        """An old scheme. it is slow. But since we only do it once, we keep it."""
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
                        # ---------------------------------------------------------
                    # -------------------------------------------------------------
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
                # ------------------------------------------------------------------
        # ------------------------------------------------------------------------------
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
