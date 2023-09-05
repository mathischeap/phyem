# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from tools.quadrature import Quadrature
from msehy.py2.mesh.elements.level.main import MseHyPy2MeshLevel
from msehy.py2.mesh.elements.visualize import MseHyPy2MeshElementsVisualize
from msehy.py2.mesh.elements.coordinate_transformation import MseHyPy2MeshElementsCoordinateTransformation
from msehy.py2.mesh.elements.fundamental_cell import MseHyPy2MeshFundamentalCell


class MseHyPy2MeshElements(Frozen):
    """"""

    def __init__(self, generation, background, region_wise_refining_strength_function, refining_thresholds):
        """"""
        self.___generation___ = generation
        self._background = background
        self._refining(region_wise_refining_strength_function, refining_thresholds)
        self._visualize = None
        self._ct = None
        self._collecting_fundamental_cells()
        self._map = None
        self._freeze()

    @property
    def background(self):
        return self._background

    def __repr__(self):
        """repr"""
        return rf"<{self.generation}th generation msehy-elements UPON {self.background}>"

    @property
    def generation(self):
        """I am the ``generation``th generation."""
        return self.___generation___

    def __iter__(self):
        """go through all fundamental cells"""
        for index in self._fundamental_cells:
            yield index

    def __len__(self):
        """How many fundamental cells?"""
        return len(self._fundamental_cells)

    def __contains__(self, index):
        """If index indicating a fundamental cell?"""
        return index in self._fundamental_cells

    def __getitem__(self, index):
        """Get the fundamental cell instance of index ``index`."""
        return self._fundamental_cells[index]

    @property
    def map(self):
        """collection of the map of all fundamental cells."""
        if self._map is None:
            self._map = dict()
            for i in self:
                self._map[i] = self[i].map
        return self._map

    def _collecting_fundamental_cells(self):
        """"""
        self._fundamental_cells = dict()
        if self.num_levels == 0:
            if self.background._elements is None:
                return
            else:
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

        self._msepy_element_indices = msepy_element_indices
        self._fundamental_triangle_indices = fundamental_triangle_indices

        # initialize fundamental cells -----------------------------------------
        for i in msepy_element_indices:
            self._fundamental_cells[i] = MseHyPy2MeshFundamentalCell(self, i)
        for i in fundamental_triangle_indices:
            self._fundamental_cells[i] = MseHyPy2MeshFundamentalCell(self, i)

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

    @property
    def visualize(self):
        if self._visualize is None:
            self._visualize = MseHyPy2MeshElementsVisualize(self)
        return self._visualize

    @property
    def ct(self):
        """ct"""
        if self._ct is None:
            self._ct = MseHyPy2MeshElementsCoordinateTransformation(self)
        return self._ct

    def _refining(self, region_wise_refining_strength_function, refining_thresholds):
        """"""
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

        assert refining_thresholds.ndim == 1 and np.alltrue(np.diff(refining_thresholds) >= 0), \
            f"refining_thresholds={refining_thresholds} is wrong, it must be a increasing 1d array."
        assert refining_thresholds[0] >= 0, \
            f"refining_thresholds={refining_thresholds} wrong, thresholds must > 0."

        # - check region_wise_refining_strength_function ----------------------------------------------------
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
        """"""
        from msehy.py2.main import __msehy_py2_setting__
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
        """"""
        from msehy.py2.main import __msehy_py2_setting__
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
        """

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
        from msehy.py2.main import __msehy_py2_setting__
        degree = __msehy_py2_setting__['refining_examining_factor']
        degree = [degree for _ in range(self.background.n)]  # must be 2
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
