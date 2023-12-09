# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from tools.quadrature import Quadrature
from tools.functions.space._2d.angle import angle as _2_angle
from tools.functions.space._2d.distance import distance as _2_distance
from legacy.msehy.py._2d.mesh.elements.level.coordinate_tranformation import MseHyPy2TrianglesCoordinateTransformation
from legacy.msehy.py._2d.mesh.elements.level.triangle import MseHyPy2MeshElementsLevelTriangle

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
        r"""Find the base element and characters of a triangle indexed `index`.

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

        Then we return 75, characters.

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

        return _global_character_cache[index]

    @property
    def local_map(self):
        return self._local_map

    def _make_local_map(self):
        r"""
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
            if self._background.abstract.manifold.is_periodic():
                self._periodic_deeper_level_local_map_maker()
            else:
                self._non_periodic_deeper_level_local_map_maker()

    def _periodic_deeper_level_local_map_maker(self):
        """"""
        self._non_periodic_deeper_level_local_map_maker()  # we first do the non-periodic one!
        local_map = self._local_map
        to_be_decided_edges = dict()
        base_msepy_elements_map = self._background.elements.map
        for index in local_map:
            _map = local_map[index]
            assert len(_map) == 3, 'must be!'
            for edge_index, mp in zip(['b', 0, 1], _map):
                if mp is None:
                    _ = index.index('=')
                    side = index[_+1]
                    if side == '2':
                        k = 0
                        j = 1
                    elif side == '0':
                        k = 1
                        j = 0
                    elif side == '3':
                        k = 2
                        j = 3
                    elif side == '1':
                        k = 3
                        j = 2
                    else:
                        raise Exception()
                    base_element = int(index[:_])

                    b_map = base_msepy_elements_map[base_element][k]

                    if b_map == -1:  # it is on a real boundary, just pase
                        pass
                    else:
                        key = ((base_element, k), (b_map, j))
                        if key not in to_be_decided_edges:
                            to_be_decided_edges[key] = list()
                        else:
                            pass
                        position = (index, edge_index)
                        to_be_decided_edges[key].append(position)

        now_triangles = self._triangle_dict

        now_anchor_points = dict()
        watching_faces = dict()
        for key in to_be_decided_edges:
            s_element_face, o_element_face = key
            for face in key:
                ele, face_id = face
                if face_id == 0:
                    _edge = '2'
                elif face_id == 1:
                    _edge = '0'
                elif face_id == 2:
                    _edge = '3'
                elif face_id == 3:
                    _edge = '1'
                else:
                    raise Exception

                str_repr = f"{face[0]}=" + _edge

                if str_repr in watching_faces:
                    watching_faces[str_repr] += 1
                else:
                    watching_faces[str_repr] = 1

            s_ele, s_face_index = s_element_face

            if s_face_index in (0, 2):
                anchor = (np.array([-1]), np.array([-1]))
            elif s_face_index == 1:
                anchor = (np.array([1]), np.array([-1]))
            elif s_face_index == 3:
                anchor = (np.array([-1]), np.array([1]))
            else:
                raise Exception()

            anchor = self._background.elements[s_ele].ct.mapping(*anchor)
            s_a_x, s_a_y = anchor
            s_a_x = s_a_x[0]
            s_a_y = s_a_y[0]

            for triangle_edge in to_be_decided_edges[key]:
                index, edge_index = triangle_edge
                if edge_index == 'b':
                    anchor = (np.array([1]), np.array([0]))
                elif edge_index == 0:
                    anchor = (np.array([0]), np.array([-1]))
                elif edge_index == 1:
                    anchor = (np.array([0]), np.array([1]))
                else:
                    raise Exception()

                # noinspection PyUnresolvedReferences
                anchor = now_triangles[index].ct.mapping(*anchor)
                _a_x, _a_y = anchor
                _a_x = _a_x[0]
                _a_y = _a_y[0]

                vector = (_a_x - s_a_x, _a_y - s_a_y)
                assert triangle_edge not in now_anchor_points, f'must be new!'
                now_anchor_points[triangle_edge] = vector

        old_triangles = self._level._base_level_elements._triangle_dict
        old_anchor_points = dict()
        for index in old_triangles:
            _index = index[:(index.index('=') + 2)]
            if _index in watching_faces:
                _f = _index[-1]
                if _f == '0':
                    anchor = (np.array([1]), np.array([-1]))
                elif _f == '1':
                    anchor = (np.array([-1]), np.array([1]))
                elif _f == '2':
                    anchor = (np.array([-1]), np.array([-1]))
                elif _f == '3':
                    anchor = (np.array([-1]), np.array([-1]))
                else:
                    raise Exception()

                ot = old_triangles[index]

                anchor = ot._base_element.ct.mapping(*anchor)
                s_a_x, s_a_y = anchor
                s_a_x = s_a_x[0]
                s_a_y = s_a_y[0]

                for j in ('b', 0, 1):
                    if j == 'b':
                        anchor = (np.array([1]), np.array([0]))
                    elif j == 0:
                        anchor = (np.array([0]), np.array([-1]))
                    elif j == 1:
                        anchor = (np.array([0]), np.array([1]))
                    else:
                        raise Exception()

                    anchor = ot.ct.mapping(*anchor)
                    _a_x, _a_y = anchor
                    _a_x = _a_x[0]
                    _a_y = _a_y[0]

                    vector = (_a_x - s_a_x, _a_y - s_a_y)

                    _save_index = (index, j)
                    assert _save_index not in old_anchor_points, f'must be new!'
                    old_anchor_points[_save_index] = vector

        for p0_p1 in to_be_decided_edges:
            positions = to_be_decided_edges[p0_p1]
            p0, p1 = p0_p1

            _ = p1[1]
            if _ == 0:
                str_ = f"{p1[0]}=2"
            elif _ == 1:
                str_ = f"{p1[0]}=0"
            elif _ == 2:
                str_ = f"{p1[0]}=3"
            elif _ == 3:
                str_ = f"{p1[0]}=1"
            else:
                raise Exception
            len_str = len(str_)
            reverse_position = (p1, p0)
            for index_edge in positions:
                index, edge_index = index_edge

                if edge_index == 'b':
                    real_edge_index = 0
                elif edge_index == 0:
                    real_edge_index = 1
                elif edge_index == 1:
                    real_edge_index = 2
                else:
                    raise Exception

                assert self.local_map[index][real_edge_index] is None, \
                    f"these locations are where we want to find correct pair."

                anchor = now_anchor_points[index_edge]

                if reverse_position in to_be_decided_edges:

                    reverse_edges = to_be_decided_edges[reverse_position]
                    for pos in reverse_edges:
                        assert pos in now_anchor_points

                        if np.allclose(anchor, now_anchor_points[pos]):
                            assert self.local_map[index][real_edge_index] is None
                            self.local_map[index][real_edge_index] = pos
                            break
                else:
                    pass

                if self.local_map[index][real_edge_index] is None:
                    for old_index in old_anchor_points:
                        if old_index[0][:len_str] == str_:

                            old_anchor = old_anchor_points[old_index]

                            if np.allclose(anchor, old_anchor):
                                # noinspection PyTypeChecker
                                self.local_map[index][real_edge_index] = old_index
                                break
                            else:
                                pass
                else:
                    pass

                assert self.local_map[index][real_edge_index] is not None, f'must pair it to someone!'

                pos0 = (index, edge_index)
                pos1 = self.local_map[index][real_edge_index]
                sign = self._parse_sign_1(pos0, pos1)
                pos1 += (sign, )
                assert len(pos1) == 3, f'must be!'
                self.local_map[index][real_edge_index] = pos1

    def _parse_sign_1(self, position0, position1):
        """"""
        positions = (position0, position1)
        vectors = list()
        for pos in positions:
            edge_index = pos[1]
            if edge_index == 'b':
                x = np.array([1, 1])
                y = np.array([-1, 1])
            elif edge_index == 0:
                x = np.array([-1, 1])
                y = np.array([-1, -1])
            elif edge_index == 1:
                x = np.array([-1, 1])
                y = np.array([1, 1])
            else:
                raise Exception()

            if pos[0].count('-') == self._level_num:
                triangle = self[pos[0]]
            else:  # position is at previous level
                triangle = self._level._base_level_elements._triangle_dict[pos[0]]

            # noinspection PyUnresolvedReferences
            x, y = triangle.ct.mapping(x, y)
            x = x[1] - x[0]
            y = y[1] - y[0]
            vectors.append((x, y))
        x0y0, x1y1 = vectors
        x0, y0 = x0y0
        x1, y1 = x1y1
        if np.isclose(x0, x1) and np.isclose(y0, y1):
            return '+'
        else:
            return '-'

    def _non_periodic_deeper_level_local_map_maker(self):
        """Silly but works.
        """
        assert self._level_num > 0, f'must be.'
        refining_triangles = self._level._refining_elements
        # since the level is larger than 0 level, must be triangles.
        all_base_triangles = self._level._base_level_elements._triangle_dict
        not_refined_triangles = list()
        for t in all_base_triangles:
            if t not in refining_triangles:
                not_refined_triangles.append(t)
            else:
                pass

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
                x = np.round(x, 7)
                y = np.round(y, 7)

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
                x = np.round(x, 7)
                y = np.round(y, 7)

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
            indicator = remaining[_]
            if _ in last_edges:
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
        _pd_ = dict()
        for index in self._local_map:
            _map = self._local_map[index]
            assert len(_map) == 3, f"must be!"
            for j, mp in zip(['b', 0, 1], _map):
                if mp is None:
                    pass
                else:

                    self_position = (index, j)

                    if isinstance(mp, tuple):
                        if mp[0].count('-') == self._level_num:
                            # at same level
                            other_position = mp[:2]

                            if other_position not in _pd_:
                                assert self_position not in _pd_, 'must be!'
                                _pd_[self_position] = other_position
                            else:
                                assert _pd_[other_position] == self_position, 'must be!'

                    elif isinstance(mp, list):
                        pass

                    else:
                        raise Exception()
