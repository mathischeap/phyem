# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from tools.quadrature import Quadrature
from tools.functions.space._2d.angle import angle as _2_angle
from tools.functions.space._2d.distance import distance as _2_distance
from msehy.py2.mesh.elements.level.coordinate_tranformation import MseHyPy2TrianglesCoordinateTransformation
from msehy.py2.mesh.elements.level.triangle import MseHyPy2MeshElementsLevelTriangle

_global_character_cache = {}


class MseHyPy2LevelTriangles(Frozen):
    """"""

    def __init__(self, level):
        """"""
        self._level = level
        self._background = level.background
        self._level_num = level._level_num
        self._make_elements()
        self._ct = MseHyPy2TrianglesCoordinateTransformation(self)
        self._make_local_map()
        self._check_local_map()
        self._freeze()

    @property
    def background(self):
        return self._background

    def __repr__(self):
        """repr"""
        return (f"<G[{self._level._elements.generation}]" +
                f"triangles of levels[{self._level_num}] UPON {self.background}>")

    @property
    def ct(self):
        """"""
        return self._ct

    def area(self, triangle_range=None):
        """"""
        quad = Quadrature([5, 5], category='Gauss')
        nodes = quad.quad_ndim[:-1]
        weights = quad.quad_ndim[-1]
        detJ = self.ct.Jacobian(*nodes, triangle_range=triangle_range)
        area = dict()
        for e in detJ:
            area[e] = np.sum(detJ[e] * weights)
        return area

    def _make_elements(self):
        """Make the elements, the triangles."""
        indices = list()
        _refining_elements = self._level._refining_elements
        if self._level_num == 0:
            # ...
            for e in _refining_elements:
                indices.extend(
                    [f"{e}=0", f"{e}=1", f"{e}=2", f"{e}=3"]
                )

        else:
            for base_index in _refining_elements:
                indices.extend(
                    [base_index+'-0', base_index+'-1']
                )

        self._triangle_dict = {}
        for ei in indices:
            self._triangle_dict[ei] = None

    def __getitem__(self, index):
        """Return the local element indexed `index` on this level."""
        assert index in self._triangle_dict, \
            f"element indexed {index} is not a valid element on level[{self._level_num}]."
        if self._triangle_dict[index] is None:
            # noinspection PyTypeChecker
            self._triangle_dict[index] = MseHyPy2MeshElementsLevelTriangle(self, index)
        else:
            pass
        return self._triangle_dict[index]

    def __contains__(self, index):
        """if triangle ``index`` is a valid local triangle."""
        return index in self._triangle_dict

    def __iter__(self):
        """iter"""
        for index in self._triangle_dict:
            yield index

    def __len__(self):
        return len(self._triangle_dict)

    def _find_characters_of_triangle(self, index):
        """Find the base element and characters of a triangle indexed `index`.

        characters = [(xt, yt), (xb, yb)] where, for example,

        ^ y
        |
        |
           . t
          /|\
         / |h\
        /__|__\
           b
        --------------> x

        The corner of vertex o is 90 degree.

        0) (xt ,yt) is the coordinates of point `t` relative to the reference element [-1,1]
        1) (xb ,yb) is the coordinates of point `b` relative to the reference element [-1,1]

        So if this is the triangle in base-element #75 on level[0], characters must be [(0, 0), (0, -1)].

        Then we return 75, characters

        """
        if index in _global_character_cache:
            return _global_character_cache[index]
        else:
            pass

        if '-' not in index:   # this triangle must be on level[0]
            base_element, sequence = index.split('=')
            base_element = int(base_element)
            top_coo = (0, 0)
            bot_coo = (-1, -1)
            match sequence:
                case '0':
                    bot_coo = (1, 0)
                case '1':
                    bot_coo = (0, 1)
                case '2':
                    bot_coo = (-1, 0)
                case '3':
                    bot_coo = (0, -1)

            characters = [top_coo, bot_coo]

        else:
            splitting = index.split('-')
            base_index = splitting[:-1]
            base_index = '-'.join(base_index)
            ind = splitting[-1]
            base_element, base_characters = self._find_characters_of_triangle(base_index)
            x0_y0, x1_y1 = base_characters
            base_angle = _2_angle(x0_y0, x1_y1)
            h2 = _2_distance(x0_y0, x1_y1) / 2
            pi2 = np.pi/2
            x0, y0 = x0_y0
            x1, y1 = x1_y1
            x = (x0 + x1) / 2
            y = (y0 + y1) / 2

            if ind == '0':
                angle = base_angle - pi2
                bx = x + h2 * np.cos(angle)
                by = y + h2 * np.sin(angle)
            elif ind == '1':
                angle = base_angle + pi2
                bx = x + h2 * np.cos(angle)
                by = y + h2 * np.sin(angle)
            else:
                raise Exception()

            characters = [x1_y1, (bx, by)]

        angle = _2_angle(*characters)
        if angle < 0:
            angle += 2 * np.pi
        else:
            pass

        degree_angle = round(angle * 180 / np.pi, 6)
        assert degree_angle in [0, 45, 90, 135, 180, 225, 270, 315, 360], \
            f"angle = {angle}, equivalent to {degree_angle} degree is impossible."

        _global_character_cache[index] = base_element, characters

        return base_element, characters

    @property
    def local_map(self):
        return self._local_map

    def _make_local_map(self):
        """
        A dict. Keys are indices. Values are a list of three entries indicating the object at the bottom
        -> edge0 -> edge1 side.

        Returns
        -------

        """
        self._local_map = dict()
        if self._level_num == 0:
            base_map = self._level.background.elements.map
            for e in self._level._refining_elements:
                surrounding_base_elements = base_map[e]

                # 'e=0' triangle
                bottom_base_element = surrounding_base_elements[1]
                if bottom_base_element == -1:
                    bottom = None
                else:
                    if bottom_base_element not in self._level._refining_elements:
                        bottom = [bottom_base_element, 0, 0, '+']
                    else:
                        bottom = (f'{bottom_base_element}=2', 'b', '-')

                self._local_map[f'{e}=0'] = [
                    bottom,
                    (f'{e}=3', 1, '+'),
                    (f'{e}=1', 0, '+'),
                ]

                # 'e=1' triangle
                bottom_base_element = surrounding_base_elements[3]
                if bottom_base_element == -1:
                    bottom = None
                else:
                    if bottom_base_element not in self._level._refining_elements:
                        bottom = [bottom_base_element, 1, 0, '-']
                    else:
                        bottom = (f'{bottom_base_element}=3', 'b', '-')

                self._local_map[f'{e}=1'] = [
                    bottom,
                    (f'{e}=0', 1, '+'),
                    (f'{e}=2', 0, '+'),
                ]

                # 'e=2' triangle
                bottom_base_element = surrounding_base_elements[0]
                if bottom_base_element == -1:
                    bottom = None
                else:
                    if bottom_base_element not in self._level._refining_elements:
                        bottom = [bottom_base_element, 0, 1, '-']
                    else:
                        bottom = (f'{bottom_base_element}=0', 'b', '-')

                self._local_map[f'{e}=2'] = [
                    bottom,
                    (f'{e}=1', 1, '+'),
                    (f'{e}=3', 0, '+'),
                ]

                # 'e=3' triangle
                bottom_base_element = surrounding_base_elements[2]
                if bottom_base_element == -1:
                    bottom = None
                else:
                    if bottom_base_element not in self._level._refining_elements:
                        bottom = [bottom_base_element, 1, 1, '+']
                    else:
                        bottom = (f'{bottom_base_element}=1', 'b', '-')

                self._local_map[f'{e}=3'] = [
                    bottom,
                    (f'{e}=2', 1, '+'),
                    (f'{e}=0', 0, '+'),
                ]

        else:
            self._deeper_level_local_map_maker()

    def _deeper_level_local_map_maker(self):
        """make element map for level 1+. This is a very silly scheme. Pls update it!"""
        assert self._level_num > 0, f'must be.'
        refining_triangles = self._level._refining_elements  # since larger than 0 level, be triangles.
        all_base_triangles = self._level._base_level_elements._triangle_dict
        not_refined_triangles = list()
        for t in all_base_triangles:
            if t not in refining_triangles:
                not_refined_triangles.append(t)

        base_ct = self._level._base_level_elements.ct

        last_edges = {}

        _edges = [
            base_ct.mapping(
                np.array([1, 1, 1]), np.array([-1, 0, 1]), triangle_range=not_refined_triangles
            ), base_ct.mapping(
                np.array([-1, 0, 1]), np.array([-1, -1, -1]), triangle_range=not_refined_triangles
            ), base_ct.mapping(
                np.array([-1, 0, 1]), np.array([1, 1, 1]), triangle_range=not_refined_triangles
            )
        ]

        for _edge, edge_index in zip(_edges, ['b', 0, 1]):
            for index in _edge:
                indicator0 = (index, edge_index, '+')
                indicator1 = (index, edge_index, '-')

                x, y = _edge[index]
                x[np.isclose(x, 0)] = 0
                y[np.isclose(y, 0)] = 0
                x = np.round(x, 5)
                y = np.round(y, 5)

                str0 = str(x) + str(y)
                str1 = str(x[::-1]) + str(y[::-1])

                if str0 in last_edges:
                    del last_edges[str0]
                else:
                    last_edges[str0] = indicator0
                if str1 in last_edges:
                    del last_edges[str1]
                else:
                    last_edges[str1] = indicator1

        _edges = [
            self.ct.mapping(
                np.array([1, 1, 1]), np.array([-1, 0, 1])
            ), self.ct.mapping(
                np.array([-1, 0, 1]), np.array([-1, -1, -1])
            ), self.ct.mapping(
                np.array([-1, 0, 1]), np.array([1, 1, 1])
            )
        ]

        for index in self:
            self._local_map[index] = [None, None, None]

        self_edges = dict()
        for _edge, edge_index, j in zip(_edges, ['b', 0, 1], [0, 1, 2]):
            for index in _edge:
                indicator0 = (index, edge_index, '+')
                indicator1 = (index, edge_index, '-')

                x, y = _edge[index]
                x[np.isclose(x, 0)] = 0
                y[np.isclose(y, 0)] = 0
                x = np.round(x, 5)
                y = np.round(y, 5)

                str0 = str(x) + str(y)
                str1 = str(x[::-1]) + str(y[::-1])

                if str0 in self_edges:
                    self_edges[str0].append(indicator0)
                else:
                    self_edges[str0] = [indicator0]

                if str1 in self_edges:
                    self_edges[str1].append(indicator1)
                else:
                    self_edges[str1] = [indicator1]

        remaining = dict()
        for _ in self_edges:
            pair = self_edges[_]

            if len(pair) == 2:
                p0, p1 = pair

                if p0[1] == 'b':
                    assert p1[1] == 'b', 'must be'
                    self._local_map[p0[0]][0] = p1[:2] + ('-', )
                    self._local_map[p1[0]][0] = p0[:2] + ('-', )

                elif p0[1] == 0:
                    assert p1[1] == 1 and p0[2] == p1[2]
                    self._local_map[p0[0]][1] = p1[:2] + ('+', )
                    self._local_map[p1[0]][2] = p0[:2] + ('+', )

                else:
                    raise Exception()

            else:
                assert len(pair) == 1, f'must be'
                remaining[_] = pair[0]

        del self_edges
        for _ in remaining:
            # print(_, '   ', remaining[_])
            indicator = remaining[_]
            if _ in last_edges:
                # print(indicator, )
                last_indicator = last_edges[_]
                assert indicator[1] == 'b', 'must be'
                sign0 = indicator[2]
                sign1 = last_indicator[2]
                if sign1 == sign0:
                    sign = '+'
                else:
                    sign = '-'

                # noinspection PyTypeChecker
                self._local_map[indicator[0]][0] = last_indicator[:2] + (sign, )
            else:   # must be on domain boundary
                pass

    def _check_local_map(self):
        """"""
        pass
