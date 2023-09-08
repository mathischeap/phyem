# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from msehy.tools.irregular_gathering_matrix import IrregularGatheringMatrix


class MseHyPy2GatheringMatrixLambda(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        self._mesh = space.mesh
        self._k = space.abstract.k
        self._orientation = space.abstract.orientation
        self._cache = dict()
        self._cache0 = {
            'gp': (-1, -1),
            'iGM': IrregularGatheringMatrix({})
        }
        self._freeze()

    def __call__(self, degree, g):
        """"""
        g = self._mesh._pg(g)
        p = self._space[degree].p
        cache_key = str(p)  # only need p
        cached, cache_gm = self._mesh.generations.sync_cache(self._cache, g, cache_key)
        if cached:
            return cache_gm
        else:
            pass
        if self._k == 1:
            method_name = f"_k{self._k}_{self._orientation}"
        else:
            method_name = f"_k{self._k}"
        iGM = getattr(self, method_name)(g, p)
        self._check_iGM(iGM, degree, g)
        self._mesh.generations.sync_cache(self._cache, g, cache_key, data=iGM)
        return iGM

    def _check_iGM(self, iGM, degree, g):
        """"""
        local_dof_coo = self._space.local_dof_representative_coordinates(degree)
        q_coo = local_dof_coo['q']
        t_coo = local_dof_coo['t']
        q_x, q_y = q_coo
        t_x, t_y = t_coo
        representative = self._mesh[g]
        dof_coo_dict = dict()
        assert len(representative) == len(iGM), f"gathering matrix length wrong."
        for i in representative:
            assert i in iGM, f"numbering for fundamental cell {i} is missed."
            gm = iGM[i]
            fc = representative[i]
            if isinstance(i, str):
                coo = np.vstack(fc.ct.mapping(t_x, t_y))
            else:
                coo = np.vstack(fc.ct.mapping(q_x, q_y))

            coo[np.isclose(coo, 0)] = 0
            coo = np.round(coo, decimals=5).T
            for c, gmc in zip(coo, gm):
                coo_str = str(c)
                if coo_str in dof_coo_dict:
                    assert gmc == dof_coo_dict[coo_str], f"global number #{gmc} appears at different places."
                else:
                    dof_coo_dict[coo_str] = gmc

    def _k0(self, g, p):
        """"""
        if (g, p) == self._cache0['gp']:
            return self._cache0['iGM']
        else:
            pass

        self._cache0['gp'] = (g, p)

        representative = self._mesh[g]
        px, py = p
        assert px == py, f"must be the case for msehy-py2"
        corner_xi = np.array([-1, 1, -1, 1])
        corner_et = np.array([-1, -1, 1, 1])
        corner_coo_q = representative.ct.mapping(corner_xi, corner_et, fc_range=representative._q_range)
        corner_xi = np.array([-1, 1, 1])
        corner_et = np.array([-1, -1, 1])
        corner_coo_t = representative.ct.mapping(corner_xi, corner_et, fc_range=representative._t_range)

        for i in corner_coo_t:
            x, y = corner_coo_t[i]
            _ = np.vstack([x, y])
            _[np.isclose(_, 0)] = 0
            _ = _.round(5)
            corners = [
                str(list(_[:, 0])),
                str(list(_[:, 1])),
                str(list(_[:, 2])),
            ]
            corner_coo_t[i] = corners
        for i in corner_coo_q:
            x, y = corner_coo_q[i]
            _ = np.vstack([x, y])
            _[np.isclose(_, 0)] = 0
            _ = _.round(5)
            corners = [
                str(list(_[:, 0])),
                str(list(_[:, 1])),
                str(list(_[:, 2])),
                str(list(_[:, 3])),
            ]
            corner_coo_q[i] = corners

        corner_dict = dict()
        for i in representative:
            if i in corner_coo_t:
                corners = corner_coo_t[i]
                for corner in corners:
                    if corner in corner_dict:
                        pass
                    else:
                        corner_dict[corner] = -1
            elif i in corner_coo_q:
                corners = corner_coo_q[i]
                for corner in corners:
                    if corner in corner_dict:
                        pass
                    else:
                        corner_dict[corner] = -1
            else:
                raise Exception()

        NUMBERING = dict()
        _map = representative.map

        q_num_basis = (px+1) * (py+1)
        t_num_basis = px * (py+1) + 1

        local_numbering = self._space.local_numbering.Lambda._k0(p)

        local_numbering_q = local_numbering['q'][0]
        q_c00 = local_numbering_q[0, 0]
        q_c10 = local_numbering_q[-1, 0]
        q_c01 = local_numbering_q[0, -1]
        q_c11 = local_numbering_q[-1, -1]
        q_e00 = local_numbering_q[0, 1:-1]
        q_e01 = local_numbering_q[-1, 1:-1]
        q_e10 = local_numbering_q[1:-1, 0]
        q_e11 = local_numbering_q[1:-1, -1]
        q_internal = local_numbering_q[1:-1, 1:-1].ravel('F')

        local_numbering_t = local_numbering['t'][0]
        t_c_top = local_numbering_t[0, 0]
        t_c0 = local_numbering_t[-1, 0]
        t_c1 = local_numbering_t[-1, -1]
        t_e_bottom = local_numbering_t[-1, 1:-1]
        t_e0 = local_numbering_t[1:-1, 0]
        t_e1 = local_numbering_t[1:-1, -1]
        t_internal = local_numbering_t[1:-1, 1:-1].ravel('F')

        num_basis_internal = (px - 1) * (py - 1)
        current = 0
        for i in representative.background.elements:
            fc_indices = list()
            if i in representative._q_range:
                fc_indices.append(i)
            else:
                look_for = f'{i}='
                len_lf = len(look_for)
                for t_i in representative._t_range:
                    if t_i[:len_lf] == look_for:
                        fc_indices.append(t_i)
                    else:
                        pass

            for fc_index in fc_indices:
                mp_i = _map[fc_index]
                assert fc_index not in NUMBERING, f"fundamental cell {fc_index} is already numbered."

                if isinstance(fc_index, int):
                    numbering_ = - np.ones(q_num_basis, dtype=int)
                    # this is a `q` fundamental cell
                    assert all([_ in corner_dict for _ in corner_coo_q[fc_index]]), f"must be!"
                    corner_00, corner_10, corner_01, corner_11 = corner_coo_q[fc_index]

                    # 1) number x-, y- corner -------------------------------------------------------------
                    if corner_dict[corner_00] == -1:
                        corner_dict[corner_00] = current
                        current += 1
                    numbering_[q_c00] = corner_dict[corner_00]

                    # 2) number y- edge -------------------------------------------------------------------
                    indicator = mp_i[2]
                    if indicator is None:
                        _nbr = np.arange(current, current + px - 1)
                        current += px - 1
                    elif isinstance(indicator, tuple):
                        target_index = indicator[0]
                        assert len(indicator) == 3 and indicator[1] == 'b', \
                            f"a base mesh element must pair to bottom of a triangle"
                        if target_index in NUMBERING:
                            sign = indicator[2]
                            _nbr = NUMBERING[target_index][t_e_bottom]
                            if sign == '-':
                                _nbr = _nbr[::-1]
                            else:
                                pass
                        else:
                            _nbr = np.arange(current, current + px - 1)
                            current += px - 1
                    else:
                        assert indicator % 1 == 0, f"must be"
                        if indicator not in NUMBERING:
                            _nbr = np.arange(current, current + px - 1)
                            current += px - 1
                        else:
                            _nbr = NUMBERING[indicator][q_e11]

                    numbering_[q_e10] = _nbr

                    # 3) number x+, y- corner -------------------------------------------------------------
                    if corner_dict[corner_10] == -1:
                        corner_dict[corner_10] = current
                        current += 1
                    numbering_[q_c10] = corner_dict[corner_10]

                    # 4) number x- edge -------------------------------------------------------------------
                    indicator = mp_i[0]
                    if indicator is None:
                        _nbr = np.arange(current, current + py - 1)
                        current += py - 1
                    elif isinstance(indicator, tuple):
                        target_index = indicator[0]
                        assert len(indicator) == 3 and indicator[1] == 'b', \
                            f"a base mesh element must pair to bottom of a triangle"
                        if target_index in NUMBERING:
                            sign = indicator[2]
                            _nbr = NUMBERING[target_index][t_e_bottom]
                            if sign == '-':
                                _nbr = _nbr[::-1]
                            else:
                                pass
                        else:
                            _nbr = np.arange(current, current + py - 1)
                            current += py - 1
                    else:
                        assert indicator % 1 == 0, f"must be"
                        if indicator not in NUMBERING:
                            _nbr = np.arange(current, current + py - 1)
                            current += py - 1
                        else:
                            _nbr = NUMBERING[indicator][q_e01]

                    numbering_[q_e00] = _nbr

                    # 5) number internal ------------------------------------------------------------
                    numbering_[q_internal] = np.arange(current, current + num_basis_internal)
                    current += num_basis_internal

                    # 6) number x+ edge -------------------------------------------------------------
                    indicator = mp_i[1]
                    if indicator is None:
                        _nbr = np.arange(current, current + py - 1)
                        current += py - 1
                    elif isinstance(indicator, tuple):
                        target_index = indicator[0]
                        assert len(indicator) == 3 and indicator[1] == 'b', \
                            f"a base mesh element must pair to bottom of a triangle"
                        if target_index in NUMBERING:
                            sign = indicator[2]
                            _nbr = NUMBERING[target_index][t_e_bottom]
                            if sign == '-':
                                _nbr = _nbr[::-1]
                            else:
                                pass
                        else:
                            _nbr = np.arange(current, current + py - 1)
                            current += py - 1
                    else:
                        assert indicator % 1 == 0, f"must be"
                        if indicator not in NUMBERING:
                            _nbr = np.arange(current, current + py - 1)
                            current += py - 1
                        else:
                            _nbr = NUMBERING[indicator][q_e00]

                    numbering_[q_e01] = _nbr

                    # 7) number x-, y+ corner -------------------------------------------------------------
                    if corner_dict[corner_01] == -1:
                        corner_dict[corner_01] = current
                        current += 1
                    numbering_[q_c01] = corner_dict[corner_01]

                    # 8) number y+ edge -------------------------------------------------------------
                    indicator = mp_i[3]
                    if indicator is None:
                        _nbr = np.arange(current, current + px - 1)
                        current += px - 1
                    elif isinstance(indicator, tuple):
                        target_index = indicator[0]
                        assert len(indicator) == 3 and indicator[1] == 'b', \
                            f"a base mesh element must pair to bottom of a triangle"
                        if target_index in NUMBERING:
                            sign = indicator[2]
                            _nbr = NUMBERING[target_index][t_e_bottom]
                            if sign == '-':
                                _nbr = _nbr[::-1]
                            else:
                                pass
                        else:
                            _nbr = np.arange(current, current + px - 1)
                            current += px - 1
                    else:
                        assert indicator % 1 == 0, f"must be"
                        if indicator not in NUMBERING:
                            _nbr = np.arange(current, current + px - 1)
                            current += px - 1
                        else:
                            _nbr = NUMBERING[indicator][q_e10]

                    numbering_[q_e11] = _nbr

                    # 9) number x+, y+ corner -------------------------------------------------------------
                    if corner_dict[corner_11] == -1:
                        corner_dict[corner_11] = current
                        current += 1
                    numbering_[q_c11] = corner_dict[corner_11]

                elif isinstance(fc_index, str):
                    numbering_ = - np.ones(t_num_basis, dtype=int)
                    # this is a `q` fundamental cell
                    assert all([_ in corner_dict for _ in corner_coo_t[fc_index]]), f"must be!"
                    corner_top, corner_0, corner_1 = corner_coo_t[fc_index]

                    # 1) number top corner -------------------------------------------------------------
                    if corner_dict[corner_top] == -1:
                        corner_dict[corner_top] = current
                        current += 1
                    numbering_[t_c_top] = corner_dict[corner_top]

                    # 2) number edge0 -------------------------------------------------------------------
                    indicator = mp_i[1]
                    if indicator is None:
                        _nbr = np.arange(current, current + px - 1)
                        current += px - 1
                    else:
                        assert isinstance(indicator, tuple)
                        target_index = indicator[0]

                        if target_index in NUMBERING:
                            i1 = indicator[1]
                            if i1 == 0:
                                _ = t_e0
                            elif i1 == 1:
                                _ = t_e1
                            else:
                                _ = t_e_bottom
                            sign = indicator[2]
                            _nbr = NUMBERING[target_index][_]
                            if sign == '-':
                                _nbr = _nbr[::-1]
                            else:
                                pass
                        else:
                            _nbr = np.arange(current, current + px - 1)
                            current += px - 1
                    numbering_[t_e0] = _nbr

                    # 3) number corner 0 -------------------------------------------------------------
                    if corner_dict[corner_0] == -1:
                        corner_dict[corner_0] = current
                        current += 1
                    numbering_[t_c0] = corner_dict[corner_0]

                    # 4) number internal -------------------------------------------------------------
                    numbering_[t_internal] = np.arange(current, current + num_basis_internal)
                    current += num_basis_internal

                    # 5) number edge bottom ----------------------------------------------------------
                    indicator = mp_i[0]
                    if indicator is None:
                        _nbr = np.arange(current, current + py - 1)
                        current += py - 1
                    elif isinstance(indicator, tuple):
                        target_index = indicator[0]

                        if target_index in NUMBERING:
                            i1 = indicator[1]
                            if i1 == 'b':
                                _ = t_e_bottom
                            elif i1 == 0:
                                _ = t_e0
                            else:
                                _ = t_e1
                            sign = indicator[2]
                            _nbr = NUMBERING[target_index][_]
                            if sign == '-':
                                _nbr = _nbr[::-1]
                            else:
                                pass
                        else:
                            _nbr = np.arange(current, current + py - 1)
                            current += py - 1

                    elif isinstance(indicator, list):
                        target_index = indicator[0]

                        if target_index in NUMBERING:
                            m, n = indicator[1:3]
                            sign = indicator[3]

                            if m == n == 0:
                                _ = q_e00
                            elif m == 0 and n == 1:
                                _ = q_e01
                            elif m == 1 and n == 0:
                                _ = q_e10
                            else:
                                _ = q_e11

                            _nbr = NUMBERING[target_index][_]
                            if sign == '-':
                                _nbr = _nbr[::-1]
                            else:
                                pass
                        else:
                            _nbr = np.arange(current, current + py - 1)
                            current += py - 1

                    else:
                        raise Exception

                    numbering_[t_e_bottom] = _nbr

                    # 6) number edge1 -------------------------------------------------------------------
                    indicator = mp_i[2]
                    if indicator is None:
                        _nbr = np.arange(current, current + px - 1)
                        current += px - 1
                    else:
                        assert isinstance(indicator, tuple)
                        target_index = indicator[0]

                        if target_index in NUMBERING:
                            i1 = indicator[1]
                            if i1 == 0:
                                _ = t_e0
                            elif i1 == 1:
                                _ = t_e1
                            else:
                                _ = t_e_bottom
                            sign = indicator[2]
                            _nbr = NUMBERING[target_index][_]
                            if sign == '-':
                                _nbr = _nbr[::-1]
                            else:
                                pass
                        else:
                            _nbr = np.arange(current, current + px - 1)
                            current += px - 1
                    numbering_[t_e1] = _nbr

                    # 7) number corner 1 -------------------------------------------------------------
                    if corner_dict[corner_1] == -1:
                        corner_dict[corner_1] = current
                        current += 1
                    numbering_[t_c1] = corner_dict[corner_1]

                else:
                    raise Exception

                assert -1 not in numbering_, f"every local dof must be numbered."

                NUMBERING[fc_index] = numbering_

        iGM = IrregularGatheringMatrix(NUMBERING)
        self._cache0['iGM'] = iGM
        return iGM

    def _k1_inner(self, g, p):
        """"""
        local_numbering = self._space.local_numbering.Lambda._k1_inner(p)

        local_numbering_q = local_numbering['q']
        local_numbering_q_dx, local_numbering_q_dy = local_numbering_q

        local_numbering_t = local_numbering['t']
        local_numbering_t_dx, local_numbering_t_dy = local_numbering_t

        return self.___k1___(
            local_numbering_q_dx, local_numbering_q_dy,
            local_numbering_t_dx, local_numbering_t_dy,
            g, p
        )

    def _k1_outer(self, g, p):
        """"""
        local_numbering = self._space.local_numbering.Lambda._k1_outer(p)

        local_numbering_q = local_numbering['q']
        local_numbering_q_dy, local_numbering_q_dx = local_numbering_q
        local_numbering_t = local_numbering['t']
        local_numbering_t_dy, local_numbering_t_dx = local_numbering_t

        return self.___k1___(
            local_numbering_q_dx, local_numbering_q_dy,
            local_numbering_t_dx, local_numbering_t_dy,
            g, p
        )

    def ___k1___(
        self,
        local_numbering_q_dx, local_numbering_q_dy,
        local_numbering_t_dx, local_numbering_t_dy,
        g, p
    ):
        representative = self._mesh[g]
        px, py = p
        assert px == py, f"must be the case for msehy-py2"

        q_num_basis = px * (py+1) + (px+1) * py
        t_num_basis = px * (py+1) + px * py

        q_dx_0 = local_numbering_q_dx[:, 0]
        q_dx_1 = local_numbering_q_dx[:, -1]
        q_dx_internal = local_numbering_q_dx[:, 1:-1].ravel('F')

        q_dy_0 = local_numbering_q_dy[0, :]
        q_dy_1 = local_numbering_q_dy[-1, :]
        q_dy_internal = local_numbering_q_dy[1:-1, :].ravel('F')

        t_dx_0 = local_numbering_t_dx[:, 0]
        t_dx_1 = local_numbering_t_dx[:, -1]
        t_dx_internal = local_numbering_t_dx[:, 1:-1].ravel('F')

        t_dy_bottom = local_numbering_t_dy[-1, :]
        t_dy_internal = local_numbering_t_dy[:-1, :].ravel('F')

        num_dx_internal = px * (py-1)
        num_dy_internal = (px-1) * py

        current = 0
        NUMBERING = dict()
        _map = representative.map

        for i in representative.background.elements:
            fc_indices = list()
            if i in representative._q_range:
                fc_indices.append(i)
            else:
                look_for = f'{i}='
                len_lf = len(look_for)
                for t_i in representative._t_range:
                    if t_i[:len_lf] == look_for:
                        fc_indices.append(t_i)
                    else:
                        pass

            for fc_index in fc_indices:
                mp_i = _map[fc_index]
                assert fc_index not in NUMBERING, f"fundamental cell {fc_index} is already numbered."

                if isinstance(fc_index, int):
                    numbering_ = - np.ones(q_num_basis, dtype=int)
                    # this is a `q` fundamental cell
                    # 1) number y- edge -------------------------------------------------------------------
                    indicator = mp_i[2]
                    if indicator is None:
                        _nbr = np.arange(current, current + px)
                        current += px
                    elif isinstance(indicator, tuple):
                        target_index = indicator[0]
                        assert len(indicator) == 3 and indicator[1] == 'b', \
                            f"a base mesh element must pair to bottom of a triangle"
                        if target_index in NUMBERING:
                            sign = indicator[2]
                            _nbr = NUMBERING[target_index][t_dy_bottom]
                            if sign == '-':
                                _nbr = _nbr[::-1]
                            else:
                                pass
                        else:
                            _nbr = np.arange(current, current + px)
                            current += px
                    else:
                        assert indicator % 1 == 0, f"must be"
                        if indicator not in NUMBERING:
                            _nbr = np.arange(current, current + px)
                            current += px
                        else:
                            _nbr = NUMBERING[indicator][q_dx_1]

                    numbering_[q_dx_0] = _nbr

                    # 2) number dx internal ------------------------------------------------------------
                    if num_dx_internal == 0:
                        pass
                    else:
                        numbering_[q_dx_internal] = np.arange(current, current + num_dx_internal)
                        current += num_dx_internal

                    # 3) number y+ edge -------------------------------------------------------------
                    indicator = mp_i[3]
                    if indicator is None:
                        _nbr = np.arange(current, current + px)
                        current += px
                    elif isinstance(indicator, tuple):
                        target_index = indicator[0]
                        assert len(indicator) == 3 and indicator[1] == 'b', \
                            f"a base mesh element must pair to bottom of a triangle"
                        if target_index in NUMBERING:
                            sign = indicator[2]
                            _nbr = NUMBERING[target_index][t_dy_bottom]
                            if sign == '-':
                                _nbr = _nbr[::-1]
                            else:
                                pass
                        else:
                            _nbr = np.arange(current, current + px)
                            current += px
                    else:
                        assert indicator % 1 == 0, f"must be"
                        if indicator not in NUMBERING:
                            _nbr = np.arange(current, current + px)
                            current += px
                        else:
                            _nbr = NUMBERING[indicator][q_dx_0]

                    numbering_[q_dx_1] = _nbr

                    # 4) number x- edge -------------------------------------------------------------------
                    indicator = mp_i[0]
                    if indicator is None:
                        _nbr = np.arange(current, current + py)
                        current += py
                    elif isinstance(indicator, tuple):
                        target_index = indicator[0]
                        assert len(indicator) == 3 and indicator[1] == 'b', \
                            f"a base mesh element must pair to bottom of a triangle"
                        if target_index in NUMBERING:
                            sign = indicator[2]
                            _nbr = NUMBERING[target_index][t_dy_bottom]
                            if sign == '-':
                                _nbr = _nbr[::-1]
                            else:
                                pass
                        else:
                            _nbr = np.arange(current, current + py)
                            current += py
                    else:
                        assert indicator % 1 == 0, f"must be"
                        if indicator not in NUMBERING:
                            _nbr = np.arange(current, current + py)
                            current += py
                        else:
                            _nbr = NUMBERING[indicator][q_dy_1]

                    numbering_[q_dy_0] = _nbr

                    # 5) number dy internal ------------------------------------------------------------
                    if num_dy_internal == 0:
                        pass
                    else:
                        numbering_[q_dy_internal] = np.arange(current, current + num_dy_internal)
                        current += num_dy_internal

                    # 6) number x+ edge -------------------------------------------------------------
                    indicator = mp_i[1]
                    if indicator is None:
                        _nbr = np.arange(current, current + py)
                        current += py
                    elif isinstance(indicator, tuple):
                        target_index = indicator[0]
                        assert len(indicator) == 3 and indicator[1] == 'b', \
                            f"a base mesh element must pair to bottom of a triangle"
                        if target_index in NUMBERING:
                            sign = indicator[2]
                            _nbr = NUMBERING[target_index][t_dy_bottom]
                            if sign == '-':
                                _nbr = _nbr[::-1]
                            else:
                                pass
                        else:
                            _nbr = np.arange(current, current + py)
                            current += py
                    else:
                        assert indicator % 1 == 0, f"must be"
                        if indicator not in NUMBERING:
                            _nbr = np.arange(current, current + py)
                            current += py
                        else:
                            _nbr = NUMBERING[indicator][q_dy_0]

                    numbering_[q_dy_1] = _nbr

                elif isinstance(fc_index, str):
                    numbering_ = - np.ones(t_num_basis, dtype=int)
                    # this is a `t` fundamental cell

                    # 1) number edge0 -------------------------------------------------------------------
                    indicator = mp_i[1]
                    if indicator is None:
                        _nbr = np.arange(current, current + px)
                        current += px
                    else:
                        assert isinstance(indicator, tuple)
                        target_index = indicator[0]

                        if target_index in NUMBERING:
                            i1 = indicator[1]
                            if i1 == 0:
                                _ = t_dx_0
                            elif i1 == 1:
                                _ = t_dx_1
                            else:
                                _ = t_dy_bottom
                            sign = indicator[2]
                            _nbr = NUMBERING[target_index][_]
                            if sign == '-':
                                _nbr = _nbr[::-1]
                            else:
                                pass
                        else:
                            _nbr = np.arange(current, current + px)
                            current += px
                    numbering_[t_dx_0] = _nbr

                    # 2) number dx internal -------------------------------------------------------------
                    if num_dx_internal == 0:
                        pass
                    else:
                        numbering_[t_dx_internal] = np.arange(current, current + num_dx_internal)
                        current += num_dx_internal

                    # 3) number edge1 -------------------------------------------------------------------
                    indicator = mp_i[2]
                    if indicator is None:
                        _nbr = np.arange(current, current + px)
                        current += px
                    else:
                        assert isinstance(indicator, tuple)
                        target_index = indicator[0]

                        if target_index in NUMBERING:
                            i1 = indicator[1]
                            if i1 == 0:
                                _ = t_dx_0
                            elif i1 == 1:
                                _ = t_dx_1
                            else:
                                _ = t_dy_bottom
                            sign = indicator[2]
                            _nbr = NUMBERING[target_index][_]
                            if sign == '-':
                                _nbr = _nbr[::-1]
                            else:
                                pass
                        else:
                            _nbr = np.arange(current, current + px)
                            current += px
                    numbering_[t_dx_1] = _nbr

                    # 4) number dy internal -------------------------------------------------------------
                    if num_dy_internal == 0:
                        pass
                    else:
                        numbering_[t_dy_internal] = np.arange(current, current + num_dy_internal)
                        current += num_dy_internal

                    # 5) number edge bottom ----------------------------------------------------------
                    indicator = mp_i[0]
                    if indicator is None:
                        _nbr = np.arange(current, current + py)
                        current += py
                    elif isinstance(indicator, tuple):
                        target_index = indicator[0]

                        if target_index in NUMBERING:
                            i1 = indicator[1]
                            if i1 == 'b':
                                _ = t_dy_bottom
                            elif i1 == 0:
                                _ = t_dx_0
                            else:
                                _ = t_dx_1
                            sign = indicator[2]
                            _nbr = NUMBERING[target_index][_]
                            if sign == '-':
                                _nbr = _nbr[::-1]
                            else:
                                pass
                        else:
                            _nbr = np.arange(current, current + py)
                            current += py

                    elif isinstance(indicator, list):
                        target_index = indicator[0]

                        if target_index in NUMBERING:
                            m, n = indicator[1:3]
                            sign = indicator[3]

                            if m == n == 0:
                                _ = q_dy_0
                            elif m == 0 and n == 1:
                                _ = q_dy_1
                            elif m == 1 and n == 0:
                                _ = q_dx_0
                            else:
                                _ = q_dx_1

                            _nbr = NUMBERING[target_index][_]
                            if sign == '-':
                                _nbr = _nbr[::-1]
                            else:
                                pass
                        else:
                            _nbr = np.arange(current, current + py)
                            current += py

                    else:
                        raise Exception

                    numbering_[t_dy_bottom] = _nbr

                else:
                    raise Exception

                assert -1 not in numbering_, f"every local dof must be numbered."

                NUMBERING[fc_index] = numbering_

        iGM = IrregularGatheringMatrix(NUMBERING)
        return iGM

    def _k2(self, g, p):
        """"""
        representative = self._mesh[g]
        px, py = p
        assert px == py
        NUMBERING = dict()
        current = 0
        local_dofs = px * py
        for i in representative.background.elements:
            fc_indices = list()
            if i in representative._q_range:
                fc_indices.append(i)
            else:
                look_for = f'{i}='
                len_lf = len(look_for)
                for t_i in representative._t_range:
                    if t_i[:len_lf] == look_for:
                        fc_indices.append(t_i)
                    else:
                        pass

            for fc_index in fc_indices:
                NUMBERING[fc_index] = np.arange(current, current+local_dofs)
                current += local_dofs

        return IrregularGatheringMatrix(NUMBERING)
