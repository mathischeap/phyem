# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
"""
import sys
if './' not in sys.path:
    sys.path.append('./')

import numpy as np
from src.tools.frozen import Frozen


class MsePyMeshElements(Frozen):
    """"""

    def __init__(self, mesh):
        """"""
        self._mesh = mesh
        self._origin = None
        self._nodes = None   # like _origin, but have 1 at the ends. So it means the distribution of grid lines.
        self._delta = None
        self._distribution = None   # how many elements along each axis?
        self._numbering = None
        self._num = None
        self._num_accumulation = None
        self._index_mapping = None
        self._map = None
        self.___layout_cache_key___ = None
        self._freeze()

    @property
    def map(self):
        return self._map

    @property
    def _layout_cache_key(self):
        """If the `layout_cache_key` of two regions are the same, we think their element layouts are the same.

        Remember, regions of the same layout_cache_key only means their layouts are the same, but their region
        metric can be totally different.
        """
        if self.___layout_cache_key___ is None:
            self.___layout_cache_key___ = dict()
            for i in self._delta:
                self.___layout_cache_key___[i] = hash(str(self._delta[i]))
        return self.___layout_cache_key___

    def _generate_elements_from_layout(self, layouts):
        """

        Parameters
        ----------
        layouts

        Returns
        -------

        """
        self._check_layouts(layouts)
        self._origin, self._delta, self._distribution, self._nodes = self._parse_origin_and_delta_from_layout(layouts)
        self._numbering, self._num, self._num_accumulation = self._generate_element_numbering_from_layout(layouts)
        self._index_mapping = self._generate_indices_mapping_from_layout(layouts)
        self._map = self._generate_element_map(layouts)

    def _check_layouts(self, layouts):
        """"""
        assert len(layouts) == len(self._mesh.manifold.regions), f"layout length wrong."
        if len(layouts) == 1:
            pass
        else:
            if self._mesh.manifold.regions.is_structured():
                region_map = self._mesh.manifold.regions.map
                for i in region_map:
                    layout_i = layouts[i]
                    for j, target in enumerate(region_map[i]):
                        if target is not None:
                            axis = j // 2
                            layout_j = layouts[target]
                            ly_i_axis = layout_i[axis]
                            ly_j_axis = layout_j[axis]
                            try:
                                np.testing.assert_array_almost_equal(ly_i_axis, ly_j_axis, decimal=5)
                            except AssertionError:
                                raise Exception(f"layout along {axis}-axis for "
                                                f"region #{i} and region #{target} does not match.")
            else:
                pass

    def _parse_origin_and_delta_from_layout(self, layouts):
        """"""
        origin = dict()
        nodes = dict()
        for i in layouts:
            assert i in self._mesh.manifold.regions
            layout = layouts[i]
            origin[i] = list()
            nodes[i] = list()
            for lyt in layout:
                origin[i].append([0, ])
                _lyt_one = lyt[0:-1]
                if len(_lyt_one) == 0:
                    pass
                else:
                    for j in range(len(_lyt_one)):
                        origin[i][-1].append(np.sum(_lyt_one[0:j+1]))

                assert round(origin[i][-1][-1] + lyt[-1], 9) == 1, f"safety check"
                nodes[i].append(origin[i][-1] + [1])
                origin[i][-1] = np.array(origin[i][-1])
                nodes[i][-1] = np.array(nodes[i][-1])
        delta = layouts
        dis = dict()
        for i in delta:
            dis[i] = [len(_) for _ in delta[i]]
        return origin, delta, dis, nodes

    def _generate_element_numbering_from_layout(self, layouts):
        """"""
        regions = self._mesh.manifold.regions
        element_numbering = dict()
        # a naive way of numbering elements.
        current_number = 0
        num_accumulation = list()
        for i in regions:
            layout_of_region = layouts[i]
            element_distribution = [len(_) for _ in layout_of_region]
            number_local_elements = np.prod(element_distribution)
            element_numbering[i] = np.arange(
                current_number, current_number+number_local_elements
            ).reshape(element_distribution, order='F')
            num_accumulation.append(current_number)
            current_number += number_local_elements
        amount_of_elements = int(current_number)
        return element_numbering, amount_of_elements, num_accumulation

    def _generate_indices_mapping_from_layout(self, layouts):
        """"""
        regions = self._mesh.manifold.regions

        existing_unique = False
        for i in regions:
            region = regions[i]
            ctm = region._ct.mtype
            indicator = ctm._indicator
            if indicator == 'Unique':
                existing_unique = True
            else:
                pass

        if existing_unique:
            mip = MsePyMeshElementsIndexMapping(self._num)
        else:

            element_mtype_dict = dict()

            for i in regions:
                layout_of_region = layouts[i]
                element_numbering_of_region = self._numbering[i]
                region = regions[i]
                ctm = region._ct.mtype
                r_emd = ctm._distribute_to_element(layout_of_region, element_numbering_of_region)
                for key in r_emd:
                    if key in element_mtype_dict:
                        element_mtype_dict[key].extend(r_emd[key])
                    else:
                        element_mtype_dict[key] = r_emd[key]

            for key in element_mtype_dict:
                element_mtype_dict[key].sort()

            mip = MsePyMeshElementsIndexMapping(element_mtype_dict, self._num)

        reference_delta = list()
        reference_origin = list()
        reference_regions = list()
        for ce in mip._reference_elements:
            in_region, local_indices = self._find_region_and_local_indices_of_element(ce)
            reference_regions.append(in_region)
            delta_s = self._delta[in_region]
            origin_s = self._origin[in_region]
            rd = list()
            ro = list()
            for i, index in enumerate(local_indices):
                rd.append(delta_s[i][index])
                ro.append(origin_s[i][index])
            reference_delta.append(rd)
            reference_origin.append(ro)
        mip._reference_delta = tuple(reference_delta)   # not need to convert to array
        mip._reference_origin = tuple(reference_origin)   # not need to convert to array
        mip._reference_regions = tuple(reference_regions)  # not need to convert to array

        return mip

    def _find_region_and_local_indices_of_element(self, i):
        """

        Parameters
        ----------
        i

        Returns
        -------

        """
        assert 0 <= i < self._num, f"i={i} wrong, I have {self._num} elements, i must be in [0, {self._num}]."
        assert i % 1 == 0, f"i must be integer."
        in_region = -1
        num_regions = len(self._mesh.manifold.regions)
        for j, na in enumerate(self._num_accumulation):
            if j == num_regions - 1:  # the last region.
                in_region = j
                break
            else:
                if na <= i < self._num_accumulation[j+1]:
                    in_region = j
                    break
                else:
                    pass

        assert in_region != -1, f"must have found a region."

        local_numbering = i - self._num_accumulation[in_region]
        # numbering = self._numbering[in_region]
        dis = self._distribution[in_region]

        ndim = len(dis)

        indices = list()
        for _ in range(ndim-1):
            n = ndim - 1 - _

            num_layer = np.prod(dis[:n])

            indices.append(local_numbering // num_layer)

            local_numbering = local_numbering % num_layer

        indices.append(local_numbering)

        indices = indices[::-1]

        return in_region, indices

    def _generate_element_map(self, layouts):
        """"""
        structured_regions = self._mesh.manifold.regions.is_structured()

        if structured_regions:  # `map_type = 0` region map. See `_check_map` of `regions`.

            element_map = self._generate_element_map_form_structured_regions(layouts)
            # return a 2-d array as the element-map

        else:
            raise NotImplementedError()
            # should return a 1-d data-structure of strings (which indicate the location of the element.)

        return element_map

    def _generate_element_map_form_structured_regions(self, layouts):
        """"""
        numbering = self._numbering
        total_num_elements = self._num
        ndim = self._mesh.ndim
        element_map = - np.ones((total_num_elements, 2 * ndim), dtype=int)

        for i in numbering:
            _nmb = numbering[i]
            layout = layouts[i]
            element_distribution = [len(_) for _ in layout]
            assert len(layout) == ndim, f"layout[{i}] is wrong."

            for axis in range(ndim):
                for layer in range(element_distribution[axis]):

                    for_elements = self._find_on_layer(_nmb, axis, layer)

                    plus_layer = layer + 1
                    minus_layer = layer - 1

                    if minus_layer < 0:
                        assert minus_layer == -1, f"safety check"
                        neighbor_region = self._find_region_on(i, axis, 0)
                        if neighbor_region is None:  # this side is boundary
                            minus_side_elements = None
                        else:
                            neighbor_region_numbering = numbering[neighbor_region]
                            minus_side_elements = self._find_on_layer(
                                neighbor_region_numbering, axis, -1
                            )

                    else:
                        minus_side_elements = self._find_on_layer(_nmb, axis, minus_layer)

                    if plus_layer >= element_distribution[axis]:
                        assert plus_layer == element_distribution[axis], f"safety check"
                        neighbor_region = self._find_region_on(i, axis, 1)
                        if neighbor_region is None:  # this side is boundary
                            plus_side_elements = None
                        else:
                            neighbor_region_numbering = numbering[neighbor_region]
                            plus_side_elements = self._find_on_layer(
                                neighbor_region_numbering, axis, 0
                            )

                    else:
                        plus_side_elements = self._find_on_layer(_nmb, axis, plus_layer)

                    if minus_side_elements is not None:
                        element_map[for_elements, 2*axis] = minus_side_elements
                    else:
                        pass
                    if plus_side_elements is not None:
                        element_map[for_elements, 2*axis + 1] = plus_side_elements
                    else:
                        pass

        return element_map

    def _find_region_on(self, i, axis, side):
        """"""
        rmp = self._mesh.manifold.regions.map[i]
        return rmp[2*axis + side]

    def _find_on_layer(self, numbering, axis, layer):
        """

        Parameters
        ----------
        numbering
        axis
        layer

        Returns
        -------

        """
        if axis == 0:
            return numbering[layer, ...]
        elif axis == 1:
            return numbering[:, layer, ...]
        elif axis == 2:
            return numbering[:, :, layer, ...]
        else:
            raise NotImplementedError()


class MsePyMeshElementsIndexMapping(Frozen):
    """"""

    def __init__(self, ci_ei_map, total_num_elements=None):
        """

        Parameters
        ----------
        ci_ei_map :
            cache_index -> element_indices

        """
        if isinstance(ci_ei_map, int):  # one to one mapping; each element is unique.
            # in this case, ci_ei_map is the amount of elements.
            ci_ei_map = np.arange(ci_ei_map)[:, np.newaxis]
            ei_ci_map = ci_ei_map[:, 0]

        elif isinstance(ci_ei_map, dict):
            ci_ei_map = tuple(ci_ei_map.values())
            ei_ci_map = np.zeros(total_num_elements, dtype=int)
            for i, indices in enumerate(ci_ei_map):
                ei_ci_map[indices] = i

        else:
            raise NotImplementedError()

        self._c2e = ci_ei_map
        self._e2c = ei_ci_map

        # these elements as representatives will be used to compute the metric for groups.
        self._reference_elements = list()
        for ce in self._c2e:
            self._reference_elements.append(ce[0])
        self._reference_elements = np.array(self._reference_elements)
        self._reference_origin = None   # the origin of the representative elements, it is initialized!
        self._reference_delta = None   # the delta of the representative elements, it is initialized!
        self._reference_regions = None
        self.___involved_regions___ = None
        self._freeze()

    @property
    def _involved_regions(self):
        """The involved regions."""
        if self.___involved_regions___ is None:
            self.___involved_regions___ = list(set(self._reference_regions))
        return self.___involved_regions___

    def distribute_according_to_reference_elements_dict(self, data_dict):
        """We make a distributor from a dict of data whose keys are the reference elements.

        Parameters
        ----------
        data_dict

        Returns
        -------

        """
        assert len(self._reference_elements) == len(data_dict)
        for re in self._reference_elements:
            assert re in data_dict, f"data_dict miss data for reference element {re}."
        return _DataDictDistributor(self, data_dict)


class _DataDictDistributor(Frozen):
    """"""

    def __init__(self, index_mapping, data_dict):
        """"""
        self._mp = index_mapping
        self._dd = data_dict
        self._freeze()

    def get_data_of_element(self, i):
        """return the data for element #i."""
        return self._dd[
            self._mp._reference_elements[
                self._mp._e2c[i]
            ]
        ]

    def __call__(self, i):
        """return the data for element #i."""
        return self.get_data_of_element(i)

    def __getitem__(self, re):
        """return the data for reference element #i."""
        return self._dd[re]

    def __iter__(self):
        """Go through all reference elements."""
        for re in self._dd:
            return re

    def __len__(self):
        """How many reference element/data."""
        return len(self._dd)


if __name__ == '__main__':
    # python msepy/mesh/elements.py
    import __init__ as ph
    space_dim = 2
    ph.config.set_embedding_space_dim(space_dim)

    manifold = ph.manifold(space_dim)
    mesh = ph.mesh(manifold)

    msepy, obj = ph.fem.apply('msepy', locals())

    mnf = obj['manifold']
    msh = obj['mesh']

    # msepy.config(mnf)('crazy', c=0.1, periodic=False, bounds=[[0, 2] for _ in range(space_dim)])
    msepy.config(mnf)('backward_step')
    msepy.config(msh)([3 for _ in range(space_dim)])
    # print(msh.elements.map)
