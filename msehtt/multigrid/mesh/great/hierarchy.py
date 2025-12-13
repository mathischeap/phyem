# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen


class MseHtt_MultiGrid_GreatMesh_Hierarchy(Frozen):
    r""""""

    def __init__(self, tgm):
        r""""""
        self._tgm = tgm
        self._hierarchy_ = None
        self._hierarchies_cache_ = {}
        self._freeze()

    def __call__(self):
        r"""Return the hierarchy."""
        if self._hierarchy_ is None:
            self._make_hierarchy_()
            assert self._hierarchy_ is not None, f"we must have made the hierarchy of the MG great mesh."
        return self._hierarchy_

    def visualize(self):
        r""""""
        raise NotImplementedError()

    def _make_hierarchy_(self):
        r"""We make self._hierarchy_ here. self._hierarchy_ is a dict. Its keys are all level indices except the
        max level.

        self._hierarchy_[lvl] returns a dictionary, for example,
            self._hierarchy_[0] = {
                0: (indicator0, tuple of element indices of the next level mesh),
                1: (indicator1, tuple of element indices of the next level mesh),
                ...
            }
            whose keys 0, 1, ... are all the element indices of the level0 mesh. And each value, for example, the
            value of self._hierarchy_[0][1]:
                (indicator1, tuple of element indices of the next level mesh)
            has two entries. The first entry, indicator1, normally is a string. It shows what kind of hierarchy
            we have for the current level element, i.e. the element #1 on the level#0 (hereafter, referred to as
            this element).
            If indicator1 is 'nested', then it means this element is split into multiple elements on the next
            lvl mesh (There is no element on the next level cover more than one element of the previous mesh).

            And then, the second entry,
                tuple of element indices of the next level mesh
            shows those elements covering this element on the next level.

        """
        tgm = self._tgm
        level_range = tgm.level_range
        num_levels = len(level_range)

        hierarchy = dict()
        for num in range(num_levels - 1):
            cur_lvl = level_range[num]
            nxt_lvl = level_range[num + 1]
            hierarchy[cur_lvl] = self._make_hierarchy_between_levels_(cur_lvl, nxt_lvl)

        self._hierarchy_ = hierarchy

    def _make_hierarchy_between_levels_(self, cur_lvl, nxt_lvl):
        r"""We return a dictionary representing the hierarchy of lvl `cur_lvl` towards `nxt_lvl`."""
        lvl_key = (cur_lvl, nxt_lvl)
        if lvl_key in self._hierarchies_cache_:
            return self._hierarchies_cache_[lvl_key]
        else:
            pass
        refining_method = self._tgm._configuration['method']
        if refining_method == 'uniform':
            hierarchy = self._make_hierarchy_between_uniform_levels_(cur_lvl, nxt_lvl)
        else:
            raise NotImplementedError(
                f"_make_hierarchy_between_levels_ for refining_method={refining_method} is not implemented.")
        # ------- check indicator of each element ------------------------------------------------------
        allowed_indicators = ('nested', )
        for e in hierarchy:
            indicator = hierarchy[e][0]
            assert indicator in allowed_indicators, \
                f"indicator = {indicator} is illegal, it must be among {allowed_indicators}"
        # ==============================================================================================
        self._hierarchies_cache_[lvl_key] = hierarchy
        return hierarchy

    def _make_hierarchy_between_uniform_levels_(self, cur_lvl, nxt_lvl):
        r"""On a uniform type of multigrid mesh,
        we return a dictionary representing the hierarchy of lvl `cur_lvl` towards `nxt_lvl`.
        """
        assert cur_lvl < nxt_lvl, \
            f"For uniform-multigrid, we can only make hierarchy for coarse mesh towards refined mesh."
        nxt_mesh = self._tgm.get_level(nxt_lvl)
        nxt_elements = nxt_mesh.elements
        nxt_element_center_points = {}
        for i in nxt_elements:
            nxt_element = nxt_elements[i]
            nxt_element_center_points[i] = nxt_element._find_element_center_coo(nxt_element.parameters)

        hierarchy = {}
        cur_mesh = self._tgm.get_level(cur_lvl)
        cur_elements = cur_mesh.elements
        for e in cur_elements:
            cur_element = cur_elements[e]
            ceg = cur_element.geometry
            _found_next_element_indices = []
            for i in nxt_element_center_points:
                if ceg._whether_contain_point_(nxt_element_center_points[i]):
                    _found_next_element_indices.append(i)
                else:
                    pass

            hierarchy[e] = ('nested', tuple(_found_next_element_indices))
            for i in _found_next_element_indices:
                del nxt_element_center_points[i]

        assert len(nxt_element_center_points) == 0, \
            (f"we must research this case. If not, we have left some elements on the next level which do not "
             f"belong to an element on the current level.")
        return hierarchy
