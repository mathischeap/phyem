# -*- coding: utf-8 -*-
r"""
"""
from typing import Dict
from tools.frozen import Frozen
from msepy.mesh.main import MsePyMesh
import numpy as np
from tools.quadrature import Quadrature

from legacy.msehy.py._2d.mesh.elements.level.main import MseHyPy2MeshLevel
from legacy.msehy.py._2d.mesh.elements.visualize import MseHyPy2MeshElementsVisualize

from legacy.generic.py._2d_unstruct.mesh.main import GenericUnstructuredMesh2D


class MseHyPy2MeshElements(Frozen):
    """"""

    def __init__(self, generation, background, region_wise_refining_strength_function, refining_thresholds):
        """"""
        self._generation = generation
        assert background.__class__ is MsePyMesh, f'Must be!'
        self._background = background
        self._refining(region_wise_refining_strength_function, refining_thresholds)
        indices = self._collecting_fundamental_cell_indices()
        self._indices = indices
        self._element_indices = None

        # --- make a generic mesh as the real body.
        type_dict, vertex_dict, vertex_coordinates, same_vertices_dict \
            = self._make_generic_element_input_dict(indices)
        self._generic = GenericUnstructuredMesh2D(  # the real body of this mesh.
            type_dict, vertex_dict, vertex_coordinates, same_vertices_dict
        )

        # --- personal properties ------------------------------------------
        self._visualize = MseHyPy2MeshElementsVisualize(self)
        self._freeze()

    @property
    def generation(self):
        """I am the ``generation``th generation."""
        return self._generation

    @property
    def background(self):
        return self._background

    def __repr__(self):
        """repr"""
        return rf"<G[{self.generation}] msehy2-elements UPON {self.background}>"

    def _refining(self, region_wise_refining_strength_function, refining_thresholds):
        """

        Parameters
        ----------
        region_wise_refining_strength_function
        refining_thresholds

        Returns
        -------

        """
        self._leveling = list()
        self._levels = list()
        bgm = self.background
        # --- parse refining_thresholds ---------------------------------------------------------------------
        if not isinstance(refining_thresholds, np.ndarray):
            refining_thresholds = np.array(refining_thresholds)
        else:
            pass
        self._thresholds = refining_thresholds

        if len(refining_thresholds) == 0:
            return

        assert refining_thresholds.ndim == 1 and all(np.diff(refining_thresholds) >= 0), \
            f"refining_thresholds={refining_thresholds} is wrong, it must be a increasing 1d array."
        assert refining_thresholds[0] >= 0, \
            f"refining_thresholds={refining_thresholds} wrong, thresholds must > 0."

        # - check region_wise_refining_strength_function ----------------------------------------------------
        if isinstance(region_wise_refining_strength_function, str):
            # it must represent a file, we read from it.
            import pickle
            from src.config import SIZE
            assert SIZE == 1, f"ph.read works for COMM.SIZE == 1. Now it is {SIZE}."
            with open(region_wise_refining_strength_function, 'rb') as inputs:
                func = pickle.load(inputs)
            inputs.close()
            region_wise_refining_strength_function = func
        else:
            pass

        if isinstance(region_wise_refining_strength_function, dict):
            pass
        elif callable(region_wise_refining_strength_function):
            func_dict = dict()
            for region in bgm.regions:
                func_dict[region] = region_wise_refining_strength_function
            region_wise_refining_strength_function = func_dict
        else:
            raise Exception(
                f"cannot accept refining_strength_function={region_wise_refining_strength_function}")
        assert isinstance(region_wise_refining_strength_function, dict), \
            f"region_wise_refining_strength_function must be a dict"
        assert (len(region_wise_refining_strength_function) == len(bgm.regions) and
                all([_ in region_wise_refining_strength_function for _ in bgm.regions])), \
            f"region_wise_refining_strength_function should be a dict whose keys cover all region indices."

        self._refining_function = region_wise_refining_strength_function

        # -- now let's do the refining and put the results in levels -----------------------------------------
        continue_refining = self._refine_background_elements(
            region_wise_refining_strength_function, refining_thresholds[0]
        )

        if continue_refining:
            # ----- do deeper level refining ------------------------------------------------------------------
            for j, threshold in enumerate(refining_thresholds[1:]):
                continue_refining = self._refine_triangle_level(
                    j+1, region_wise_refining_strength_function, threshold
                )
                if continue_refining:
                    pass
                else:
                    break

    def _refine_background_elements(self, func, threshold):
        """Make the first level based on the background msepy elements."""
        from legacy.msehy.py._2d.main import __msehy_py2_setting__
        scheme = __msehy_py2_setting__['refining_examining_scheme']
        elements_to_be_refined = self._examining(
            self.background.elements, None, 0, func, threshold, scheme=scheme,
        )

        if len(elements_to_be_refined) > 0:
            self._leveling.append(
                elements_to_be_refined
            )
            self._levels.append(
                MseHyPy2MeshLevel(self, 0, self.background.elements, elements_to_be_refined)
            )
            return 1
        else:
            return 0

    def _refine_triangle_level(self, level_num, func, threshold):
        """Refine levels >= 1."""
        from legacy.msehy.py._2d.main import __msehy_py2_setting__
        scheme = __msehy_py2_setting__['refining_examining_scheme']
        newest_leve = self._levels[-1].triangles
        triangle_range = list()
        for t in newest_leve:
            triangle = newest_leve[t]
            p2 = triangle.pair_to
            if p2 is None:
                triangle_range.append(t)
            elif isinstance(p2, str) and p2.count('-') == triangle._index.count('-'):
                triangle_range.append(t)
            else:
                pass

        triangles_to_be_refined = self._examining(
            newest_leve, triangle_range, 1, func, threshold, scheme=scheme,
        )
        pair_triangles_extension = list()
        for i in triangles_to_be_refined:
            triangle = newest_leve[i]
            p2 = triangle.pair_to
            if isinstance(p2, str):
                assert p2.count('-') == triangle._index.count('-'), f'must be this case'
                pair_triangles_extension.append(p2)
        triangles_to_be_refined.extend(pair_triangles_extension)
        triangles_to_be_refined = set(triangles_to_be_refined)
        triangles_to_be_refined = list(triangles_to_be_refined)
        triangles_to_be_refined.sort()

        if len(triangles_to_be_refined) > 0:
            self._leveling.append(
                triangles_to_be_refined
            )
            self._levels.append(
                MseHyPy2MeshLevel(self, level_num, newest_leve, triangles_to_be_refined)
            )
            return 1
        else:
            return 0

    def _examining(
            self,
            elements_or_triangles,
            element_or_triangle_range,
            element_type,
            func,
            threshold,
            scheme=0
    ):
        r"""Determine which cells could be further refined.

        Parameters
        ----------
        elements_or_triangles : base elements or level triangles
        element_or_triangle_range
        func
        threshold
        scheme

        Returns
        -------

        """
        from legacy.msehy.py._2d.main import __msehy_py2_setting__
        degree = __msehy_py2_setting__['refining_examining_factor']
        degree = [degree for _ in range(self.background.n)]  # self.background.n must be 2 in this case
        quad = Quadrature(degree, category='Gauss')
        nodes = quad.quad_ndim[:-1]
        weights = quad.quad_ndim[-1]

        if element_type == 0:
            xyz = elements_or_triangles.ct.mapping(*nodes, element_range=element_or_triangle_range)
            detJ = elements_or_triangles.ct.Jacobian(*nodes, element_range=element_or_triangle_range)
            area = elements_or_triangles.area(element_range=element_or_triangle_range)   # 2d: area
        elif element_type == 1:  # they are triangles Now
            xyz = elements_or_triangles.ct.mapping(*nodes, triangle_range=element_or_triangle_range)
            detJ = elements_or_triangles.ct.Jacobian(*nodes, triangle_range=element_or_triangle_range)
            area = elements_or_triangles.area(triangle_range=element_or_triangle_range)  # 2d: area
        else:
            raise Exception()

        elements_or_triangles_to_be_refined = list()
        if scheme == 0:  # a := int(abs(strength function)) / element_area, if a >= threshold, do refining.
            for e in xyz:
                x_y_z = xyz[e]
                det_J = detJ[e]
                region = elements_or_triangles[e].region
                fun = func[region]
                integration = np.sum(np.abs(fun(*x_y_z)) * det_J * weights)
                mean = integration / area[e]
                if mean >= threshold:
                    elements_or_triangles_to_be_refined.append(e)
        else:
            raise NotImplementedError()

        return elements_or_triangles_to_be_refined

    @property
    def visualize(self):
        """visualize."""
        return self._visualize

    @property
    def leveling(self):
        """the levels indicating a refining."""
        return self._leveling

    @property
    def levels(self):
        """the levels indicating a refining."""
        return self._levels

    @property
    def max_levels(self):
        """max levels"""
        return len(self._thresholds)

    @property
    def thresholds(self):
        """thresholds."""
        return self._thresholds

    @property
    def num_levels(self):
        """the amount of valid levels. Because maybe, for no refinement is made for some high thresholds."""
        return len(self.levels)

    def _collecting_fundamental_cell_indices(self):
        """"""
        self._fundamental_cells = dict()
        if self.num_levels == 0:
            assert self.background._elements is not None
            base_element_num = self.background.elements._num
            msepy_element_indices = range(base_element_num)
            fundamental_triangle_indices = set()
        else:
            fundamental_triangle_indices = set()
            exclude = set()
            for i in range(self.num_levels)[::-1]:
                lvl = self.levels[i]
                triangles = lvl.triangles
                lvl_indices = triangles._triangle_dict.keys()
                fundamental_indices = lvl_indices - exclude
                fundamental_triangle_indices.update(fundamental_indices)
                exclude = set(lvl._refining_elements)

            fundamental_triangle_indices = list(fundamental_triangle_indices)
            fundamental_triangle_indices.sort(
                key=lambda x: int(x.split('=')[0])
            )
            base_element_num = self.background.elements._num
            msepy_element_indices = set(range(base_element_num))
            msepy_element_indices = msepy_element_indices - exclude
            msepy_element_indices = list(msepy_element_indices)
            msepy_element_indices.sort()

        good_sequence = list()
        for i in self.background.elements:
            if i in msepy_element_indices:
                fc_indices = [i]
            else:
                look_for = f'{i}='
                len_lf = len(look_for)
                str_indices = list()
                for t_i in fundamental_triangle_indices:
                    if t_i[:len_lf] == look_for:
                        str_indices.append(t_i)
                    else:
                        pass
                str_indices.sort(
                    key=lambda x: x.split('=')[1]
                )
                fc_indices = str_indices

            good_sequence.extend(fc_indices)

        return good_sequence

    def _make_generic_element_input_dict(self, indices):
        """"""
        q_corners = (
            np.array([-1, 1, 1, -1]),
            np.array([-1, -1, 1, 1]),
        )
        t_corners = (
            np.array([-1, 1, 1]),
            np.array([-1, -1, 1]),
        )
        vertex_numbering = dict()
        vertex_coordinates = dict()
        current = 0

        type_dict: Dict = dict()
        vertex_dict: Dict = dict()

        type_0 = 'rt'
        type_1 = 'rq'

        for index in indices:
            if isinstance(index, str):  # a triangle cell
                level_num = index.count('-')
                triangle = self.levels[level_num].triangles[index]
                vertices = triangle.ct.mapping(*t_corners)
                x, y = vertices
                x[np.isclose(x,  0)] = 0
                y[np.isclose(y,  0)] = 0
                x = np.round(x, 7)
                y = np.round(y, 7)

                _triangle_vertex = list()
                for vx, vy in zip(x, y):
                    coo = (vx, vy)
                    vertex_key = str(vx) + ' ' + str(vy)

                    if vertex_key in vertex_numbering:
                        pass
                    else:
                        vertex_numbering[vertex_key] = current
                        assert current not in vertex_coordinates, 'must be'
                        vertex_coordinates[current] = coo
                        current += 1

                    _triangle_vertex.append(vertex_numbering[vertex_key])

                base_element = triangle._base_element
                bms = base_element.metric_signature
                if isinstance(bms, str) and bms[:6] == 'Linear':
                    type_dict[index] = type_0   # regular triangle
                    vertex_dict[index] = _triangle_vertex
                else:
                    raise NotImplementedError('distorted triangle type!')

            else:
                quadrilateral = self.background.elements[index]
                vertices = quadrilateral.ct.mapping(*q_corners)
                x, y = vertices
                x[np.isclose(x,  0)] = 0
                y[np.isclose(y,  0)] = 0
                x = np.round(x, 7)
                y = np.round(y, 7)

                _quadrilateral_vertex = list()
                for vx, vy in zip(x, y):
                    coo = (vx, vy)
                    vertex_key = str(vx) + ' ' + str(vy)

                    if vertex_key in vertex_numbering:
                        pass
                    else:
                        vertex_numbering[vertex_key] = current
                        assert current not in vertex_coordinates, 'must be'
                        vertex_coordinates[current] = coo
                        current += 1

                    _quadrilateral_vertex.append(vertex_numbering[vertex_key])

                bms = quadrilateral.metric_signature
                if isinstance(bms, str) and bms[:6] == 'Linear':
                    type_dict[index] = type_1   # regular quadrilateral
                    vertex_dict[index] = _quadrilateral_vertex
                else:
                    raise NotImplementedError('distorted quadrilateral type!')

        background = self.background
        if background.manifold.abstract._is_periodic:
            same_vertices_dict = self._find_periodic_same_vertices(vertex_numbering)

        else:
            same_vertices_dict = {}
            # for example:
            # same_vertices_dict = {
            #    11: 3,
            #    99: 3,
            #    123: 3,
            #    ...
            # }
            # this means, vertices, 11, 99, 123 are in fact the same vertex (on a periodic boundary) as vertex 3.
            # we shall use this information to make the element map.

        assert len(vertex_coordinates) == len(vertex_numbering), 'must be'
        return type_dict, vertex_dict, vertex_coordinates, same_vertices_dict

    @property
    def indices_in_base_element(self):
        """keys are base element indices, values are indices of local cells in the base elements."""
        if self._element_indices is None:
            element_indices = dict()
            for index in self._indices:
                if isinstance(index, int):
                    in_element = index
                elif isinstance(index, str):  # index referring to a triangle.
                    in_element = int(index.split('=')[0])
                else:
                    raise Exception()

                if in_element not in element_indices:
                    element_indices[in_element] = list()
                else:
                    pass
                element_indices[in_element].append(index)
            self._element_indices = element_indices
        return self._element_indices

    def _find_periodic_same_vertices(self, vertex_numbering):
        """"""
        elements = self.background.elements

        element_indices = self.indices_in_base_element

        periodic_base_edge_pairs = elements._find_periodic_pairs()

        edge_centers = (
            [np.array([-1]), np.array([0])],
            [np.array([1]), np.array([0])],
            [np.array([0]), np.array([-1])],
            [np.array([0]), np.array([1])]
        )

        q_corners = (
            np.array([-1, 1, 1, -1]),
            np.array([-1, -1, 1, 1]),
        )

        t_corners = (
            np.array([-1, 1, 1]),
            np.array([-1, -1, 1]),
        )

        same_pairs = list()

        for edge0 in periodic_base_edge_pairs:
            edge1 = periodic_base_edge_pairs[edge0]
            ele0, edge_index0 = edge0
            ele1, edge_index1 = edge1

            anchor0 = elements[ele0].ct.mapping(*edge_centers[edge_index0])
            ax0, ay0 = anchor0
            ax0 = round(ax0[0], 7)
            ay0 = round(ay0[0], 7)

            anchor1 = elements[ele1].ct.mapping(*edge_centers[edge_index1])
            ax1, ay1 = anchor1
            ax1 = round(ax1[0], 7)
            ay1 = round(ay1[0], 7)

            check_indices0 = element_indices[ele0]
            check_indices1 = element_indices[ele1]

            characteristic_vectors_0 = dict()
            for index in check_indices0:
                if isinstance(index, str):  # a triangle cell
                    level_num = index.count('-')
                    triangle = self.levels[level_num].triangles[index]
                    vertices = triangle.ct.mapping(*t_corners)
                    x, y = vertices
                    x[np.isclose(x, 0)] = 0
                    y[np.isclose(y, 0)] = 0
                    x = np.round(x, 7)
                    y = np.round(y, 7)
                    x0, x1, x2 = x
                    y0, y1, y2 = y

                    num0 = vertex_numbering[str(x0) + ' ' + str(y0)]
                    num1 = vertex_numbering[str(x1) + ' ' + str(y1)]
                    num2 = vertex_numbering[str(x2) + ' ' + str(y2)]

                    vec0 = np.array([x0 - ax0, y0 - ay0])
                    vec1 = np.array([x1 - ax0, y1 - ay0])
                    vec2 = np.array([x2 - ax0, y2 - ay0])

                    if num0 in characteristic_vectors_0:
                        assert np.allclose(characteristic_vectors_0[num0], vec0)
                    else:
                        characteristic_vectors_0[num0] = vec0

                    if num1 in characteristic_vectors_0:
                        assert np.allclose(characteristic_vectors_0[num1], vec1)
                    else:
                        characteristic_vectors_0[num1] = vec1

                    if num2 in characteristic_vectors_0:
                        assert np.allclose(characteristic_vectors_0[num2], vec2)
                    else:
                        characteristic_vectors_0[num2] = vec2

                else:
                    quadrilateral = self.background.elements[index]
                    vertices = quadrilateral.ct.mapping(*q_corners)
                    x, y = vertices
                    x[np.isclose(x, 0)] = 0
                    y[np.isclose(y, 0)] = 0
                    x = np.round(x, 7)
                    y = np.round(y, 7)

                    x0, x1, x2, x3 = x
                    y0, y1, y2, y3 = y

                    num0 = vertex_numbering[str(x0) + ' ' + str(y0)]
                    num1 = vertex_numbering[str(x1) + ' ' + str(y1)]
                    num2 = vertex_numbering[str(x2) + ' ' + str(y2)]
                    num3 = vertex_numbering[str(x3) + ' ' + str(y3)]

                    vec0 = np.array([x0 - ax0, y0 - ay0])
                    vec1 = np.array([x1 - ax0, y1 - ay0])
                    vec2 = np.array([x2 - ax0, y2 - ay0])
                    vec3 = np.array([x3 - ax0, y3 - ay0])

                    if num0 in characteristic_vectors_0:
                        assert np.allclose(characteristic_vectors_0[num0], vec0)
                    else:
                        characteristic_vectors_0[num0] = vec0

                    if num1 in characteristic_vectors_0:
                        assert np.allclose(characteristic_vectors_0[num1], vec1)
                    else:
                        characteristic_vectors_0[num1] = vec1

                    if num2 in characteristic_vectors_0:
                        assert np.allclose(characteristic_vectors_0[num2], vec2)
                    else:
                        characteristic_vectors_0[num2] = vec2

                    if num3 in characteristic_vectors_0:
                        assert np.allclose(characteristic_vectors_0[num3], vec3)
                    else:
                        characteristic_vectors_0[num3] = vec3

            characteristic_vectors_1 = dict()
            for index in check_indices1:
                if isinstance(index, str):  # a triangle cell
                    level_num = index.count('-')
                    triangle = self.levels[level_num].triangles[index]
                    vertices = triangle.ct.mapping(*t_corners)
                    x, y = vertices
                    x[np.isclose(x, 0)] = 0
                    y[np.isclose(y, 0)] = 0
                    x = np.round(x, 7)
                    y = np.round(y, 7)
                    x0, x1, x2 = x
                    y0, y1, y2 = y

                    num0 = vertex_numbering[str(x0) + ' ' + str(y0)]
                    num1 = vertex_numbering[str(x1) + ' ' + str(y1)]
                    num2 = vertex_numbering[str(x2) + ' ' + str(y2)]

                    vec0 = np.array([x0 - ax1, y0 - ay1])
                    vec1 = np.array([x1 - ax1, y1 - ay1])
                    vec2 = np.array([x2 - ax1, y2 - ay1])

                    if num0 in characteristic_vectors_1:
                        assert np.allclose(characteristic_vectors_1[num0], vec0)
                    else:
                        characteristic_vectors_1[num0] = vec0

                    if num1 in characteristic_vectors_1:
                        assert np.allclose(characteristic_vectors_1[num1], vec1)
                    else:
                        characteristic_vectors_1[num1] = vec1

                    if num2 in characteristic_vectors_1:
                        assert np.allclose(characteristic_vectors_1[num2], vec2)
                    else:
                        characteristic_vectors_1[num2] = vec2

                else:
                    quadrilateral = self.background.elements[index]
                    vertices = quadrilateral.ct.mapping(*q_corners)
                    x, y = vertices
                    x[np.isclose(x, 0)] = 0
                    y[np.isclose(y, 0)] = 0
                    x = np.round(x, 7)
                    y = np.round(y, 7)

                    x0, x1, x2, x3 = x
                    y0, y1, y2, y3 = y

                    num0 = vertex_numbering[str(x0) + ' ' + str(y0)]
                    num1 = vertex_numbering[str(x1) + ' ' + str(y1)]
                    num2 = vertex_numbering[str(x2) + ' ' + str(y2)]
                    num3 = vertex_numbering[str(x3) + ' ' + str(y3)]

                    vec0 = np.array([x0 - ax1, y0 - ay1])
                    vec1 = np.array([x1 - ax1, y1 - ay1])
                    vec2 = np.array([x2 - ax1, y2 - ay1])
                    vec3 = np.array([x3 - ax1, y3 - ay1])

                    if num0 in characteristic_vectors_1:
                        assert np.allclose(characteristic_vectors_1[num0], vec0)
                    else:
                        characteristic_vectors_1[num0] = vec0

                    if num1 in characteristic_vectors_1:
                        assert np.allclose(characteristic_vectors_1[num1], vec1)
                    else:
                        characteristic_vectors_1[num1] = vec1

                    if num2 in characteristic_vectors_1:
                        assert np.allclose(characteristic_vectors_1[num2], vec2)
                    else:
                        characteristic_vectors_1[num2] = vec2

                    if num3 in characteristic_vectors_1:
                        assert np.allclose(characteristic_vectors_1[num3], vec3)
                    else:
                        characteristic_vectors_1[num3] = vec3

            for vertex0 in characteristic_vectors_0:
                vec0 = characteristic_vectors_0[vertex0]
                for vertex1 in characteristic_vectors_1:
                    vec1 = characteristic_vectors_1[vertex1]
                    if np.allclose(vec0, vec1):
                        same_pairs.append((vertex0, vertex1))

        same_vertices_dict = dict()

        for same_pair in same_pairs:
            ver0, ver1 = same_pair
            existing = False
            for _ in same_vertices_dict:
                existing_set = same_vertices_dict[_]
                if ver0 in existing_set:
                    existing_set.add(ver1)
                    existing = True
                    break
                elif ver1 in existing_set:
                    existing_set.add(ver0)
                    existing = True
                    break
                else:
                    pass

            if existing:
                pass
            else:
                num_set = len(same_vertices_dict)
                same_vertices_dict[num_set] = {ver0, ver1}

        final_same_vertices_dict = {}
        for _ in same_vertices_dict:
            same_vertices = list(same_vertices_dict[_])
            char_ver = same_vertices[0]

            for ver in same_vertices[1:]:
                final_same_vertices_dict[ver] = char_ver

        return final_same_vertices_dict

    @property
    def generic(self):
        """The generic representative."""
        return self._generic
