# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class MseHyPy2MeshFundamentalCell(Frozen):
    """"""

    def __init__(self, elements, index):
        """"""
        self._elements = elements
        self._i = index
        self._map = None
        self._metric_signature = None
        self._ct = None
        self._freeze()

    def __repr__(self):
        """"""
        return f"<fundamental cell {self._i} of {self._elements}>"

    @property
    def representative(self):
        """It presents a triangle or a basic msepy element (quadrilateral)? Return the object!"""
        if isinstance(self._i, str):
            return self._elements.levels[self.level_num].triangles[self.index]
        else:
            return self._elements.background.elements[self.index]

    @property
    def index(self):
        """
        when it is a str, it is a triangle. When it is an int, it is a basic msepy element (quadrilateral).

        Returns
        -------

        """
        return self._i

    @property
    def level_num(self):
        if isinstance(self._i, str):
            return self._i.count('-')
        else:
            return None

    @property
    def map(self):
        """The cell map of this cell."""
        if self._map is None:
            if isinstance(self._i, str):

                level = self._elements.levels[self.level_num]
                self._map = level.triangles.local_map[self._i]

            else:  # msepy element

                raw_map = list(self._elements.background.elements.map[self._i])

                for j, obj in enumerate(raw_map):
                    if obj == -1:
                        raw_map[j] = None
                    elif obj in self._elements._msepy_element_indices:
                        pass
                    else:
                        if j == 0:
                            raw_map[j] = (f'{obj}=0', 'b', '+')
                        elif j == 1:
                            raw_map[j] = (f'{obj}=2', 'b', '-')
                        elif j == 2:
                            raw_map[j] = (f'{obj}=1', 'b', '-')
                        elif j == 3:
                            raw_map[j] = (f'{obj}=3', 'b', '+')
                        else:
                            raise Exception
                self._map = raw_map
        return self._map

    @property
    def metric_signature(self):
        """metric-signature"""
        if self._metric_signature is None:
            if isinstance(self._i, str):
                level = self._elements.levels[self.level_num]
                triangle = level.triangles[self._i]
                self._metric_signature = triangle.metric_signature
            else:
                element = self._elements.background.elements[self._i]
                element_metric_signature = element.metric_signature
                if element_metric_signature is None:
                    self._metric_signature = id(self)
                else:
                    self._metric_signature = element_metric_signature

        return self._metric_signature

    @property
    def ct(self):
        """ct"""
        if self._ct is None:
            if isinstance(self._i, str):
                level = self._elements.levels[self.level_num]
                triangle = level.triangles[self._i]
                self._ct = triangle.ct
            else:
                element = self._elements.background.elements[self._i]
                self._ct = element.ct
        return self._ct
