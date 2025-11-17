# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.tools.frozen import Frozen
from phyem.msehtt.static.manifold.predefined.distributor import Predefined_Msehtt_Manifold_Distributor


class MseHtt_Static_PreDefined_Config(Frozen):
    """"""

    def __init__(self, indicator):
        """"""
        self._regions_maker = Predefined_Msehtt_Manifold_Distributor.defined_manifolds()[indicator]
        self._freeze()

    def __call__(self, element_layout, **kwargs):
        """"""
        regions, region_map, periodic_setting = self._regions_maker(**kwargs)
        # For example, in 2d,
        # topologically, a region is mapped from (r, s) = [0, 1]^2
        #
        #   ^ s
        #   |
        #   |
        #   |  node 3        fac3 2         node 2
        #   |     -----------------------------
        #   |     |                           |
        #   |     |                           |
        #   |     |                           |
        #   |     | face 3                    | face 1
        #   |     |                           |
        #   |     |                           |
        #   |     |                           |
        #   |     |                           |
        #   |     -----------------------------
        #   |   node 0         face 0       node 1
        #   |
        #   ------------------------------------------> r
        #
        # `regions` is a dictionary of region instances. A region instance can be of any class, but it must have
        # a property `ndim` saying its dimensions and a property called `etype` saying the element
        # made in this region can only be of which type.
        #
        # `region_map` is None or a dictionary.
        # When it is a dictionary, its keys are the same to those of `regions` and values
        # are list of region corners. For example, in 2d
        #     region_map = {
        #         0: [0, 1, 2, 3],
        #         1: [1, 4, 5, 2],
        #         2: ....
        #     }
        #
        # means the numbering for four corners of region #0 are 0, 1, 2, 3, and
        # the numbering for four corners of region #1 are 1, 4, 5, 3, and so on.
        #
        # So we know the face 1 of region #0 is attached to face 3 of region #1.
        #
        for i in regions:
            region = regions[i]
            assert hasattr(region, 'ndim') and hasattr(region, 'etype'), \
                f"a valid msehtt region must have 'ndim' and 'etype' properties."
            assert hasattr(region, 'mapping') and hasattr(region, 'Jacobian_matrix'), \
                f"a valid msehtt region must have 'mapping' and 'Jacobian_matrix' methods."

        element_layout = _study_element_layout_(regions, element_layout)

        if region_map is None:
            # since this region map parser cannot identify, for example, cracks, for such
            # domains, we should provide region map to clearly clarify that.
            region_map = _parse_regions(regions)
        else:
            pass

        region_map = _parse_pbc(regions, region_map, periodic_setting)

        element_type_dict, element_parameter_dict, element_map_dict = _parse_elements_(
            regions, region_map, element_layout
        )

        return element_type_dict, element_parameter_dict, element_map_dict


def _study_element_layout_(regions, element_layout):
    """

    Parameters
    ----------
    regions:
        A dict of region instances. The region instance can be of any class, but it must have
        a property `ndim` saying its dimensions and a property called `etype` saying the element
        made in this region can only be of which type.
    element_layout

    Returns
    -------

    """
    if not isinstance(element_layout, dict):
        _temp = dict()
        for i in regions:
            _temp[i] = element_layout
        element_layout = _temp
    else:
        pass

    layout = dict()
    assert isinstance(element_layout, dict), f"element_layout must eventually be parsed as a dict!"
    for i in element_layout:  # element layout for #i region.
        layout_i = element_layout[i]
        if isinstance(layout_i, int):
            layout_i = [layout_i for _ in range(regions[i].ndim)]
        else:
            pass

        assert layout_i is not None and len(layout_i) == regions[i].ndim, \
            f"element_layout for region #{i} = {layout_i} is illegal; the region is {regions[i].ndim}-d."

        _temp = list()
        for j, layout_ij in enumerate(layout_i):
            if isinstance(layout_ij, (int, float)):
                assert layout_ij % 1 == 0 and layout_ij >= 1, \
                    f"element_layout of region #{i} = {layout_i} is illegal."
                layout_ij = np.array([1/layout_ij for i in range(int(layout_ij))])

            else:
                assert np.ndim(layout_ij) == 1, \
                    f"element_layout of region #{i} = {layout_i} is illegal."
                # noinspection PyTypeChecker
                for _ in layout_ij:
                    assert isinstance(_, (int, float)) and _ > 0, \
                        f"element_layout of region #{i} = {layout_i} is illegal."
                layout_ij = np.array(layout_ij) * 1.
                layout_ij /= np.sum(layout_ij)

            assert np.round(np.sum(layout_ij), 10) == 1, \
                f"scale layout array into 1 pls, now it is {np.sum(layout_ij)}."
            _temp.append(layout_ij)

        layout[i] = _temp

    return layout


def _parse_regions(regions):
    """"""
    Ndim = None

    region_map = {}
    # if ndim == 2, region_map['region_index'] is a list of four int.
    # if ndim == 3, region_map['region_index'] is a list of six int.
    # So, we only have topologically quadrilateral region or hexahedral region.

    region_corner_numbering_pool = {}
    current = 0
    for region_index in regions:
        region = regions[region_index]
        # each region is a class of mapping and Jacobian_matrix methods.
        # it mappings the reference region, [0,1]^d, d in (2, 3), into a real region.

        if Ndim is None:
            Ndim = region.ndim
        else:
            assert Ndim == region.ndim, f"regions are of different dimensions."

        if Ndim == 2:
            quad_sequence_region_reference_coo = (
                np.array([0, 1, 1, 0]),
                np.array([0, 0, 1, 1])
            )
        elif Ndim == 3:
            raise NotImplementedError()
        else:
            raise Exception()

        COO = region.mapping(*quad_sequence_region_reference_coo)

        if Ndim == 2:
            x0, x1, x2, x3 = COO[0]
            y0, y1, y2, y3 = COO[1]

            nodes = [
                r"%.7f-%.7f" % (x0, y0),
                r"%.7f-%.7f" % (x1, y1),
                r"%.7f-%.7f" % (x2, y2),
                r"%.7f-%.7f" % (x3, y3),
            ]
        elif Ndim == 3:
            x0, x1, x2, x3, x4, x5, x6, x7 = COO[0]
            y0, y1, y2, y3, y4, y5, y6, y7 = COO[1]
            z0, z1, z2, z3, z4, z5, z6, z7 = COO[1]

            nodes = [
                r"%.7f-%.7f-%.7f" % (x0, y0, z0),
                r"%.7f-%.7f-%.7f" % (x1, y1, z1),
                r"%.7f-%.7f-%.7f" % (x2, y2, z2),
                r"%.7f-%.7f-%.7f" % (x3, y3, z3),
                r"%.7f-%.7f-%.7f" % (x4, y4, z4),
                r"%.7f-%.7f-%.7f" % (x5, y5, z5),
                r"%.7f-%.7f-%.7f" % (x6, y6, z6),
                r"%.7f-%.7f-%.7f" % (x7, y7, z7),
            ]
        else:
            raise Exception()

        for node in nodes:
            if node in region_corner_numbering_pool:
                pass
            else:
                region_corner_numbering_pool[node] = current
                current += 1

        region_map[region_index] = list()
        if Ndim == 2:
            for i in range(4):
                node_str = nodes[i]
                region_map[region_index].append(region_corner_numbering_pool[node_str])
        elif Ndim == 3:
            for i in range(8):
                node_str = nodes[i]
                region_map[region_index].append(region_corner_numbering_pool[node_str])
        else:
            raise Exception()

    return region_map


def _parse_pbc(regions, region_map, periodic_setting):
    """We renew the region_map with periodic setting.
    """
    if periodic_setting is None:
        return region_map
    else:
        # a region cannot be periodic to itself! this is important!

        ndim = list()
        for region_index in regions:
            region = regions[region_index]
            ndim.append(region.ndim)
        assert all([ndim[0] == n for n in ndim]), f"regions must of same dimensions."
        ndim = ndim[0]
        same_labeling = []
        same_labeling_pool = set()
        if ndim == 2:
            for pos0 in periodic_setting:
                pos1 = periodic_setting[pos0]
                region0, face0 = pos0
                region1, face1 = pos1
                face0_start, face0_end = face0
                face1_start, face1_end = face1

                old_label_face0_stt = region_map[region0][face0_start]
                old_label_face0_end = region_map[region0][face0_end]

                old_label_face1_stt = region_map[region1][face1_start]
                old_label_face1_end = region_map[region1][face1_end]

                assert old_label_face0_stt != old_label_face1_stt, r"A region cannot periodic to itself"
                assert old_label_face0_end != old_label_face1_end, r"A region cannot periodic to itself"

                _ = [old_label_face0_stt, old_label_face1_stt]
                _.sort()
                same_labeling.append(_)
                same_labeling_pool.update(_)
                _ = [old_label_face0_end, old_label_face1_end]
                _.sort()
                same_labeling.append(_)
                same_labeling_pool.update(_)

        else:
            raise NotImplementedError()

        same_labeling_pool = list(same_labeling_pool)
        same_labeling_pool.sort()

        organized_same_labelling = {}
        found_label = set()
        for i in same_labeling_pool:
            if i in found_label:
                pass
            else:
                found_label.add(i)
                SAME = {i}
                for _ in range(3):  # do it for three times to find all links.
                    for pair in same_labeling:
                        p0, p1 = pair
                        if p0 in SAME and p1 not in SAME:
                            SAME.add(p1)
                            found_label.add(p1)
                        elif p1 in SAME and p0 not in SAME:
                            SAME.add(p0)
                            found_label.add(p0)
                        else:
                            pass
                organized_same_labelling[i] = SAME

        new_labelling = {}
        for n, i in enumerate(organized_same_labelling):
            new_labelling[str(n)] = organized_same_labelling[i]

        for r in region_map:
            _map = region_map[r]
            for k, j in enumerate(_map):
                if j in same_labeling_pool:
                    new_label = ''
                    for str_j in new_labelling:
                        if j in new_labelling[str_j]:
                            new_label = str_j
                            break
                        else:
                            pass
                    assert new_label != '', f"we must have found a str representation of j."
                    _map[k] = new_label
                else:
                    pass

        current = 0
        renumbering = {}
        for r in region_map:
            _map = region_map[r]
            for k, j in enumerate(_map):
                if j not in renumbering:
                    renumbering[j] = current
                    current += 1
                else:
                    pass
                _map[k] = renumbering[j]

        return region_map


def _parse_elements_(regions, region_map, element_layout):
    r""""""
    element_type_dict, element_parameter_dict, element_map_dict_str = {}, {}, {}

    Ndim = None
    current = 0

    for region_index in regions:
        region = regions[region_index]
        region_nodes = region_map[region_index]
        layout = element_layout[region_index]

        if Ndim is None:
            Ndim = region.ndim
        else:
            assert Ndim == region.ndim, f"regions are of different dimensions."

        num_elements = [len(_) for _ in layout]
        etype = region.etype

        # ==================== 2d elements ======================================================
        if Ndim == 2:
            assert etype in (9, 'unique curvilinear quad'), \
                (f"To initialize an unstructured msehtt quad mesh, we can only use `quad` or "
                 f"'unique curvilinear quadrilateral' elements.")
            I, J = num_elements
            x_spacing = list()
            y_spacing = list()
            for i in range(I):
                x_spacing.append(float(sum(layout[0][:i])))
            for j in range(J):
                y_spacing.append(float(sum(layout[1][:j])))
            x_spacing.append(1)
            y_spacing.append(1)

            for j in range(J):
                for i in range(I):
                    element_type_dict[current] = etype
                    x0 = x_spacing[i]
                    x1 = x_spacing[i + 1]
                    x2 = x_spacing[i + 1]
                    x3 = x_spacing[i]
                    y0 = y_spacing[j]
                    y1 = y_spacing[j]
                    y2 = y_spacing[j + 1]
                    y3 = y_spacing[j + 1]

                    element_nodes = (
                        (x0, y0),
                        (x1, y1),
                        (x2, y2),
                        (x3, y3),
                    )

                    element_map_dict_str[current] = [
                        ___parse_2d_element_map___(region_index, region_nodes, _) for _ in element_nodes
                    ]

                    if etype == 9:
                        element_parameter_dict[current] = [
                            region.mapping(*_) for _ in element_nodes
                        ]

                    elif etype == 'unique curvilinear quad':
                        element_CT = _RegionElementCT_2d_(region, x0, x1, y0, y2)
                        element_parameter_dict[current] = {
                            'mapping': element_CT.mapping,                  # [-1, 1]^2 into the element
                            'Jacobian_matrix': element_CT.Jacobian_matrix   # JM of mapping [-1, 1]^2 into the element
                        }

                    else:
                        raise Exception(f"2d msehtt mesh only hase quad or unique curvilinear quad elements.")

                    current += 1

        # ==================== 3d elements ======================================================
        elif Ndim == 3:
            raise NotImplementedError()

        # =======================================================================================
        else:
            raise Exception()

    element_map_dict = {}
    current = 0
    element_corner_numbering_pool = {}
    for ele in element_map_dict_str:
        element_map_dict[ele] = list()
        for corner_index in element_map_dict_str[ele]:
            if corner_index in element_corner_numbering_pool:
                pass
            else:
                element_corner_numbering_pool[corner_index] = current
                current += 1
            element_map_dict[ele].append(element_corner_numbering_pool[corner_index])

    return element_type_dict, element_parameter_dict, element_map_dict


def ___parse_2d_element_map___(region_index, region_nodes, element_node):
    r"""
    [-1, 1]^2 into the element.

    """
    x, y = element_node
    n0, n1, n2, n3 = region_nodes
    assert len(set(region_nodes)) == len(region_nodes), f"numbering of corners of a region must be different!"
    if x == y == 0:
        return n0
    elif x == 1 and y == 0:
        return n1
    elif x == 1 and y == 1:
        return n2
    elif x == 0 and y == 1:
        return n3
    elif x in (0, 1) or y in (0, 1):

        if x == 0:  # on face node0 <-> node3
            face_nodes = [n0, n3]
            dis0 = y
            dis1 = 1 - y
        elif x == 1:  # on face node1 <-> node2
            face_nodes = [n1, n2]
            dis0 = y
            dis1 = 1 - y
        elif y == 0:  # on face node0 <-> node1
            face_nodes = [n0, n1]
            dis0 = x
            dis1 = 1 - x
        elif y == 1:  # on face node3 <-> node2
            face_nodes = [n3, n2]
            dis0 = x
            dis1 = 1 - x
        else:
            raise Exception()

        fn0, fn1 = face_nodes
        if fn0 < fn1:
            return fn0, fn1, "%.7f" % dis0
        elif fn0 > fn1:
            return fn1, fn0, "%.7f" % dis1
        else:
            raise Exception(f"region node numbering must be different.")

    else:
        return region_index, '%.7f' % x, '-%.7f' % y


class _RegionElementCT_2d_(Frozen):
    r""""""
    def __init__(self, region, xlb, xub, ylb, yub):
        r""""""
        self._region = region
        self._x0 = xlb
        self._delta_x = xub - xlb
        self._y0 = ylb
        self._delta_y = yub - ylb
        self._freeze()

    def mapping(self, xi, et):
        r""""""
        r = (xi + 1) * 0.5 * self._delta_x + self._x0
        s = (et + 1) * 0.5 * self._delta_y + self._y0
        return self._region.mapping(r, s)

    def Jacobian_matrix(self, xi, et):
        r""""""
        r = (xi + 1) * 0.5 * self._delta_x + self._x0
        s = (et + 1) * 0.5 * self._delta_y + self._y0

        X, Y = self._region.Jacobian_matrix(r, s)
        xr, xs = X
        yr, ys = Y

        r_xi = 0.5 * self._delta_x
        s_et = 0.5 * self._delta_y

        x_xi = xr * r_xi
        x_et = xs * s_et

        y_xi = yr * r_xi
        y_et = ys * s_et

        return (
            [x_xi, x_et],
            [y_xi, y_et],
        )
