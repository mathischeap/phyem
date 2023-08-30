# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from tools.quadrature import Quadrature
from msehy.py2.mesh.elements.level.main import MseHyPy2MeshElementsLevel


class MseHyPy2MeshElements(Frozen):
    """"""

    def __init__(self, mesh, region_wise_refining_strength_function, refining_thresholds):
        """"""
        self._mesh = mesh
        self._refining(region_wise_refining_strength_function, refining_thresholds)
        self._freeze()

    def __repr__(self):
        """repr"""
        return rf"<{self.generation}th generation elements of {self._mesh}>"

    @property
    def generation(self):
        """I am the ``generation``th generation."""
        return self.mesh.___generation___

    @property
    def mesh(self):
        return self._mesh

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
        return len(self._thresholds)

    @property
    def num_levels(self):
        """the amount of valid levels. Because maybe, for no refinement is made for some high thresholds."""
        raise NotImplementedError()

    def _refining(self, region_wise_refining_strength_function, refining_thresholds):
        """"""
        self._leveling = list()
        self._levels = list()
        bgm = self.mesh.background
        # --- parse refining_thresholds ---------------------------------------------------------------------
        if not isinstance(refining_thresholds, np.ndarray):
            refining_thresholds = np.array(refining_thresholds)
        else:
            pass
        self._thresholds = refining_thresholds

        if len(refining_thresholds) == 0:
            return

        assert refining_thresholds.ndim == 1 and np.alltrue(np.diff(refining_thresholds) > 0), \
            f"refining_thresholds={refining_thresholds} is wrong, it must be a increasing 1d array."
        assert refining_thresholds[0] > 0, \
            f"refining_thresholds={refining_thresholds} wrong, thresholds must > 0."

        # - check region_wise_refining_strength_function ----------------------------------------------------
        assert isinstance(region_wise_refining_strength_function, dict), \
            f"region_wise_refining_strength_function must be a dict"
        assert (len(region_wise_refining_strength_function) == len(bgm.regions) and
                all([_ in region_wise_refining_strength_function for _ in bgm.regions])), \
            f"region_wise_refining_strength_function should be a dict whose keys cover all region indices."

        # -- now let's do the refining and put the results in levels -----------------------------------------
        self._refine_background_elements(region_wise_refining_strength_function, refining_thresholds[0])

    def _refine_background_elements(self, func, threshold):
        """"""
        from msehy.py2.main import __setting__
        scheme = __setting__['refining_examining_scheme']
        elements_to_be_refined = self._examining(
            self.mesh.background.elements, None, func, threshold, scheme=scheme,
        )
        self._leveling.append(
            elements_to_be_refined
        )
        self._levels.append(
            MseHyPy2MeshElementsLevel(self, 0, self.mesh.background.elements, elements_to_be_refined)
        )

    def _examining(self, elements, element_range, func, threshold, scheme=0):
        """"""
        from msehy.py2.main import __setting__
        degree = __setting__['refining_examining_factor']
        degree = [degree for _ in range(self.mesh.background.n)]  # must be 2
        quad = Quadrature(degree, category='Gauss')
        nodes = quad.quad_ndim[:-1]
        weights = quad.quad_ndim[-1]
        xyz = elements.ct.mapping(*nodes, element_range=element_range)
        detJ = elements.ct.Jacobian(*nodes, element_range=element_range)
        elements_to_be_refined = list()
        area = elements.area(element_range=element_range)  # since this is for 2-d, it must be area.
        if scheme == 0:  # a := int(strength function) / element_area, if a >= threshold, do refining.
            for e in xyz:
                x_y_z = xyz[e]
                det_J = detJ[e]
                region = elements[e].region
                fun = func[region]
                integration = np.sum(np.abs(fun(*x_y_z)) * det_J * weights)
                mean = integration / area[e]
                if mean >= threshold:
                    elements_to_be_refined.append(e)
        else:
            raise NotImplementedError()

        return elements_to_be_refined
