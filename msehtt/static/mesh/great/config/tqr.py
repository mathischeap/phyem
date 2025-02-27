# -*- coding: utf-8 -*-
r"""
All regions are rectangles.
"""
from tools.frozen import Frozen
from src.config import RANK, MASTER_RANK


class TriQuadRegions(Frozen):
    r""""""
    def __init__(
            self,
            region_type_dict,
            region_vortex_coo_dict,
            region_vortex_map,
            periodic_setting=None
    ):
        r""""""
        assert isinstance(region_vortex_map, dict), f"region_vortex_map must be a dict."
        assert isinstance(region_vortex_coo_dict, dict), f"region_vortex_coo_dict must be a dict."

        if periodic_setting is None:
            pass
        else:
            region_type_dict, region_vortex_map, region_vortex_coo_dict = (
                self._parse_periodic_setting(region_type_dict, region_vortex_map, region_vortex_coo_dict))

        self._check(region_type_dict, region_vortex_map, region_vortex_coo_dict)
        self._region_type_dict = region_type_dict
        self._region_vortex_coo_dict = region_vortex_coo_dict
        self._region_vortex_map = region_vortex_map
        self._freeze()

    @staticmethod
    def _check(region_type_dict, region_vortex_map, region_vortex_coo_dict):
        r""""""
        for r in region_type_dict:
            rm = region_vortex_map[r]
            rt = region_type_dict[r]
            if rt == 5:
                assert len(rm) == 3, f"triangle region has three vortices."
            elif rt == 9:
                assert len(rm) == 4, f"quad region has four vortices."
            else:
                raise Exception()
            for vortex in rm:
                assert vortex in region_vortex_coo_dict, f"vortex {vortex} is missing in coo dict."
                coo = region_vortex_coo_dict[vortex]
                assert len(coo) == 2, f"this is 2d region, coordinates must have two components."

    @staticmethod
    def _parse_periodic_setting(region_type_dict, region_vortex_map, region_vortex_coo_dict):
        r""""""
        raise NotImplementedError()

    def visualize(self):
        r""""""
        if RANK == MASTER_RANK:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_aspect('equal')
            for r in self._region_type_dict:
                vortices = self._region_vortex_map[r]
                x = list()
                y = list()
                for v in vortices:
                    COO = self._region_vortex_coo_dict[v]
                    x.append(COO[0])
                    y.append(COO[1])
                center_x = sum(x) / len(x)
                center_y = sum(y) / len(y)
                plt.text(center_x, center_y, r, va='center', ha='center')
                # noinspection PyUnresolvedReferences
                x.append(x[0])
                # noinspection PyUnresolvedReferences
                y.append(y[0])
                plt.plot(x, y, linewidth=0.75, c='k')
            plt.show()
            plt.close(fig)
        else:
            pass


class MseHtt_TQR_config(Frozen):
    r""""""
    def __init__(self, qr):
        r""""""
        assert qr.__class__ is TriQuadRegions, f"I must config a '{TriQuadRegions}' instance."
        self._td = qr._region_type_dict
        self._cd = qr._region_vortex_coo_dict
        self._md = qr._region_vortex_map
        self._freeze()

    def __call__(self, element_layout):
        r""""""
        if isinstance(element_layout, int) and element_layout >= 0:
            return self._element_layout_equivalent_to_ts_factor(element_layout)
        else:
            raise NotImplementedError()

    def _element_layout_equivalent_to_ts_factor(self, ts_factor):
        r"""The element_layout is a positive integer. So we do a triangular splitting using this integer as the
         factor.
         """
        element_type_dict = {}
        element_parameter_dict = {}
        element_map_dict = {}

        td = self._td
        cd = self._cd
        md = self._md

        numbering = 0
        POOL = {}

        for v in cd:
            if v in POOL:
                pass
            else:
                POOL[v] = numbering
                numbering += 1

        for r in td:
            element_type_dict[r] = td[r]
            MAP = list()
            para = list()
            original_map = md[r]
            for v in original_map:
                MAP.append(POOL[v])
                para.append(cd[v])
            element_map_dict[r] = MAP
            element_parameter_dict[r] = para

        element_type_dict, element_parameter_dict, element_map_dict = self._parse_qtr_ts_(
            ts_factor,
            element_type_dict,
            element_parameter_dict,
            element_map_dict,
        )

        return element_type_dict, element_parameter_dict, element_map_dict

    def _parse_qtr_ts_(self, ts, element_type_dict, element_parameter_dict, element_map_dict):
        r""""""
        if ts == 0:
            return element_type_dict, element_parameter_dict, element_map_dict
        else:
            element_type_dict, element_parameter_dict, element_map_dict = self.___qtr_ts___(
                element_type_dict, element_parameter_dict, element_map_dict
            )
            new_ts = ts - 1
            return self._parse_qtr_ts_(
                new_ts, element_type_dict, element_parameter_dict, element_map_dict
            )

    def ___qtr_ts___(self, element_type_dict, element_parameter_dict, element_map_dict):
        r""""""
        new_element_type_dict = {}
        new_element_parameter_dict = {}
        new_element_map_dict = {}
        for e_index in element_type_dict:
            e_type = element_type_dict[e_index]
            e_para = element_parameter_dict[e_index]
            e_map = element_map_dict[e_index]

            if e_type == 5:  # vtk-5: triangle
                A, B, C = e_para
                Ax, Ay = A
                Bx, By = B
                Cx, Cy = C
                D = ((Ax + Bx + Cx) / 3, (Ay + By + Cy) / 3)
                E = ((Ax + Bx) / 2, (Ay + By) / 2)
                F = ((Cx + Bx) / 2, (Cy + By) / 2)
                G = ((Ax + Cx) / 2, (Ay + Cy) / 2)
                _0_, _1_, _2_ = e_map
                _01_ = [_0_, _1_]
                _12_ = [_1_, _2_]
                _02_ = [_0_, _2_]
                _01_.sort()
                _12_.sort()
                _02_.sort()
                _01_ = tuple(_01_)
                _12_ = tuple(_12_)
                _02_ = tuple(_02_)
                _012_ = (_0_, _1_, _2_)
                element_index = str(e_index)
                e_i_0 = element_index + ':5>0'
                e_i_1 = element_index + ':5>1'
                e_i_2 = element_index + ':5>2'
                new_element_type_dict[e_i_0] = 9
                new_element_type_dict[e_i_1] = 9
                new_element_type_dict[e_i_2] = 9
                new_element_parameter_dict[e_i_0] = [A, E, D, G]
                new_element_parameter_dict[e_i_1] = [B, F, D, E]
                new_element_parameter_dict[e_i_2] = [C, G, D, F]
                new_element_map_dict[e_i_0] = [_0_, _01_, _012_, _02_]
                new_element_map_dict[e_i_1] = [_1_, _12_, _012_, _01_]
                new_element_map_dict[e_i_2] = [_2_, _02_, _012_, _12_]

            elif e_type == 9:  # vtk-9: quad
                #         A(node0)     J      (node3)
                #           --------------------- D
                #           |         |         |
                #           |         |E        |
                #         F |-------------------|H
                #           |         |         |
                #           |         |         |
                #         B --------------------- C
                #        (node1)      G         (node2)
                A, B, C, D = e_para
                Ax, Ay = A
                Bx, By = B
                Cx, Cy = C
                Dx, Dy = D
                E = ((Ax + Bx + Cx + Dx) / 4, (Ay + By + Cy + Dy) / 4)
                F = ((Ax + Bx) / 2, (Ay + By) / 2)
                G = ((Bx + Cx) / 2, (By + Cy) / 2)
                H = ((Cx + Dx) / 2, (Cy + Dy) / 2)
                J = ((Dx + Ax) / 2, (Dy + Ay) / 2)
                _0_, _1_, _2_, _3_ = e_map
                _01_ = [_0_, _1_]
                _12_ = [_1_, _2_]
                _23_ = [_2_, _3_]
                _30_ = [_3_, _0_]
                _01_.sort()
                _12_.sort()
                _23_.sort()
                _30_.sort()
                _01_ = tuple(_01_)
                _12_ = tuple(_12_)
                _23_ = tuple(_23_)
                _30_ = tuple(_30_)
                _0123_ = (_0_, _1_, _2_, _3_)
                element_index = str(e_index)
                e_i_0 = element_index + ':9>0'
                e_i_1 = element_index + ':9>1'
                e_i_2 = element_index + ':9>2'
                e_i_3 = element_index + ':9>3'
                new_element_type_dict[e_i_0] = 9
                new_element_type_dict[e_i_1] = 9
                new_element_type_dict[e_i_2] = 9
                new_element_type_dict[e_i_3] = 9
                new_element_parameter_dict[e_i_0] = [A, F, E, J]
                new_element_parameter_dict[e_i_1] = [F, B, G, E]
                new_element_parameter_dict[e_i_2] = [E, G, C, H]
                new_element_parameter_dict[e_i_3] = [J, E, H, D]
                new_element_map_dict[e_i_0] = [_0_, _01_, _0123_, _30_]
                new_element_map_dict[e_i_1] = [_01_, _1_, _12_, _0123_]
                new_element_map_dict[e_i_2] = [_0123_, _12_, _2_, _23_]
                new_element_map_dict[e_i_3] = [_30_, _0123_, _23_, _3_]

            else:
                raise NotImplementedError(
                    f'triangle/tetrahedron-split does not work for etype:{e_type} yet.')

        POOL = {}
        current = 0
        for r in new_element_map_dict:
            MAP = new_element_map_dict[r]
            for map_index in MAP:
                if map_index in POOL:
                    pass
                else:
                    POOL[map_index] = current
                    current += 1

        NEW_element_map_dict = {}
        for r in new_element_map_dict:
            old_map = new_element_map_dict[r]
            new_map = list()
            for old_index in old_map:
                new_index = POOL[old_index]
                new_map.append(new_index)
            NEW_element_map_dict[r] = new_map

        return new_element_type_dict, new_element_parameter_dict, NEW_element_map_dict
