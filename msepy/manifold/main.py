# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
created at: 3/30/2023 7:02 PM
"""
from importlib import import_module
from tools.frozen import Frozen
from src.config import get_embedding_space_dim

from src.manifold import NullManifold

from msepy.manifold.regions.main import MseManifoldRegions
from msepy.manifold.coordinate_transformation import MsePyManifoldsCoordinateTransformation
from msepy.manifold.visualize.main import MsePyManifoldVisualize

from msepy.manifold.regions.standard.main import MsePyManifoldStandardRegion
from msepy.manifold.regions.standard.ct import MsePyStandardRegionCoordinateTransformation

from msepy.manifold.regions.boundary.main import MsePyManifoldBoundaryRegion


def config(mf, arg, *args, **kwargs):
    """"""
    assert mf.__class__ is MsePyManifold, f"I can only config MseManifold instance. Now I get {mf}."
    assert mf._regions is None, f"manifold {mf} already have region configurations. Change it may lead to error."

    if isinstance(arg, str):  # use predefined mappings, this leads to region map type: 0

        predefined_path = '.'.join(str(MsePyManifold).split(' ')[1][1:-2].split('.')[:-2]) + \
                          '.predefined.' + arg
        _module = import_module(predefined_path)
        region_map, mapping_dict, Jacobian_matrix_dict, mtype_dict = getattr(_module, arg)(mf, **kwargs)

        mf._parse_regions_from_region_map(
            0,
            region_map,
            mapping_dict,
            Jacobian_matrix_dict,
            mtype_dict
        )
        assert mf.regions.map is not None, f"predefined manifold only config manifold with region map."
        assert mf.regions._map_type == 0, f"must be!"

    elif arg.__class__ is MsePyManifold:  # this leads to region map type: 1

        # by doing this, we config a manifold being the (a part of) the boundary of this given manifold: arg

        base_manifold = arg

        boundary_dict = args[0]

        mf._parse_regions_from_boundary_dict(base_manifold, boundary_dict)

        assert mf.regions._map_type == 1, f"must be!"

    else:
        raise NotImplementedError(f"config arg = {arg} is not understandable.")

    assert mf.regions is not None, f"we need to set regions for the manifold by this `config` function."


class MsePyManifold(Frozen):
    """"""

    def __init__(self, abstract_manifold):
        """"""
        self._abstract = abstract_manifold
        self._ct = None
        self._regions = None
        self._visualize = None
        self._freeze()

    def _parse_regions_from_region_map(
            self,
            region_map_type,
            region_map,
            mapping_dict,
            Jacobian_matrix_dict,
            mtype_dict
    ):
        """Will set `regions._map` and `region._regions` and `regions._map_type`.

        regions map type: 0

        And if it has a boundary, will parse the boundary manifold as well.
        """
        assert self._regions is None, f"Change regions will be dangerous!"
        self._regions = MseManifoldRegions(self)  # initialize the regions.
        for i in region_map:

            mapping = mapping_dict[i]

            if Jacobian_matrix_dict is None:
                Jacobian_matrix = None
            else:
                Jacobian_matrix = Jacobian_matrix_dict[i]

            if mtype_dict is None:
                mtype = None
            else:
                mtype = mtype_dict[i]

            rct = MsePyStandardRegionCoordinateTransformation(mapping, Jacobian_matrix, mtype)
            region = MsePyManifoldStandardRegion(self.regions, i, rct)
            self.regions._regions[i] = region

        self.regions._map = region_map  # ***
        self._regions._map_type = region_map_type
        assert self.abstract._is_periodic is self.regions._is_periodic(), f"Periodicity does not match."

        boundary_manifold = self.abstract.boundary()
        from src.manifold import NullManifold
        if boundary_manifold.__class__ is NullManifold:
            assert self.abstract._is_periodic, f"A manifold has null boundary must be periodic."
        else:
            from msepy.main import base
            manifolds = base['manifolds']
            the_msepy_boundary_manifold = None
            for sym in manifolds:
                msepy_manifold = manifolds[sym]
                if msepy_manifold.abstract is boundary_manifold:

                    the_msepy_boundary_manifold = msepy_manifold
                    break

                else:
                    pass

            if the_msepy_boundary_manifold is None:
                # did not find the msepy boundary manifold, which means it is not used at all, just skip then.
                pass
            else:
                boundary_dict = dict()
                r_map = self.regions.map
                for i in self.regions:
                    map_i = r_map[i]
                    boundary_dict[i] = [0 for _ in range(len(map_i))]
                    for j, mp in enumerate(map_i):
                        if mp is None:
                            boundary_dict[i][j] = 1

                the_msepy_boundary_manifold._parse_regions_from_boundary_dict(self, boundary_dict)

        self.regions._check_map()

    def _parse_regions_from_boundary_dict(self, base_manifold, boundary_dict):
        """regions map type: 1

        Parameters
        ----------
        base_manifold
        boundary_dict: dict
            {
                0: [0, 0, 1, 0, 1, ...],
                1: [1, 0, 0, ...],
                ...
            }

            It means this boundary manifold covers the y- face of region #0, z- face of region #0, ..., and
            x- face of region #1, ..., and ...

        Returns
        -------

        """
        assert self._regions is None, f"regions must be empty! Changing it is dangerous."
        # --- first check whether the boundary_dict is valid ----------------------
        assert base_manifold.regions.is_structured(), \
            f"cannot parse regions from boundary dict based on structured manifold regions (have `map`)."
        base_region_maps = base_manifold.regions.map
        for i in boundary_dict:
            assert i in base_region_maps, f"base manifold has no region #{i} at all!"
            map_i = base_region_maps[i]
            boundaries = boundary_dict[i]
            if len(boundaries) == len(map_i):
                pass
            # elif len(boundaries) < len(map_i):
            #     boundaries.extend([0 for _ in range(len(map_i) - len(boundaries))])
            else:
                raise Exception(f"boundary value {boundaries} shape wrong for region #{i}.")

            for j, bd in enumerate(boundaries):
                if bd == 0:
                    pass
                else:
                    assert bd == 1, f"boundary dict can only have 0 or 1 in the list for each base region."
                    mp = map_i[j]
                    assert mp is None, f"{j}th side of region #{i} is not a domain boundary!"

        # ------ boundary dict is OK, let's move on --------------
        self._regions = MseManifoldRegions(self)

        regions = self.regions
        regions._map_type = 1  # type 1
        regions._base_regions = base_manifold.regions
        regions._map = boundary_dict

        base_regions = base_manifold.regions
        num_regions = 0
        for i in boundary_dict:
            boundaries = boundary_dict[i]
            for j, bd in enumerate(boundaries):
                if bd == 1:
                    regions._regions[num_regions] = MsePyManifoldBoundaryRegion(
                        num_regions,
                        regions,
                        base_regions[i],
                        j
                    )
                    num_regions += 1
        self.regions._check_map()

        # ---- the manifold is configured, lets check if there is a complementary manifold can be determined ---
        abstract_base_manifold = base_manifold.abstract
        base_boundary = abstract_base_manifold.boundary()
        if base_boundary.__class__ is NullManifold:
            pass
        else:
            base_boundary_partitions = base_boundary._partitions
            for partition_num in base_boundary_partitions:
                partition = base_boundary_partitions[partition_num]
                if self.abstract in partition:

                    if len(partition) == 1:  # I am covering the whole boundary of the base manifold.

                        self._check_boundary_dict_is_full(self.regions._base_regions, [self.regions])

                    else:
                        configured_sections = list()
                        not_configured_sections = list()

                        from msepy.main import base
                        all_msepy_manifolds = base['manifolds']

                        for abs_section in partition:

                            section = None
                            for sym in all_msepy_manifolds:
                                msepy_manifold = all_msepy_manifolds[sym]
                                if msepy_manifold.abstract is abs_section:
                                    section = msepy_manifold
                                    break
                                else:
                                    pass
                            assert section is not None, \
                                f"we must have make msepy manifolds for all abstract manifolds in prior."

                            if section._regions is None:
                                not_configured_sections.append(section)
                            else:
                                configured_sections.append(section)

                        assert self in configured_sections, f'Must be this case.'

                        if len(not_configured_sections) == 0:

                            boundary_regions = list()
                            for section in configured_sections:
                                assert section.regions._map_type == 1, \
                                    f"all boundary sections must be of the same map type."
                                boundary_regions.append(section.regions)

                            self._check_boundary_dict_is_full(
                                self.regions._base_regions,
                                boundary_regions,
                            )

                        elif len(not_configured_sections) == 1:
                            boundary_region_maps = list()
                            for cfg_section in configured_sections:
                                boundary_region_maps.append(
                                    cfg_section.regions._map
                                )

                            remaining_region_sides = self._merge_boundary_dict(
                                boundary_region_maps
                            )

                            not_configured_sections[0]._parse_regions_from_boundary_dict(
                                base_manifold, remaining_region_sides
                            )

                        else:
                            pass
                else:
                    pass
        assert self.regions._map_type == 1, f'Must be the case.'

    @staticmethod
    def _check_boundary_dict_is_full(base_regions, boundary_regions):
        """check if `boundary_dict` covers all boundaries of the regions represented by `region_map`."""
        # firstly, merge boundary_dicts into one ----------------------------------
        for b_regions in boundary_regions:
            assert b_regions._base_regions is base_regions, f"boundary regions and base regions dis-match."

        base_region_map = base_regions.map
        boundary_dict = {}
        for i in base_region_map:
            boundary_dict[i] = list()
            for j, mp in enumerate(base_region_map[i]):
                bds = list()

                for _ in boundary_regions:
                    if i in _.map:
                        bds.append(_.map[i][j])
                    else:
                        bds.append(0)

                if all([_ == 0 for _ in bds]):
                    boundary_dict[i].append(0)
                else:
                    assert bds.count(1) == 1 and bds.count(0) == len(bds) - 1
                    boundary_dict[i].append(1)

        # then we check its compatibility with base manifold region map -------------
        for i in base_region_map:
            for j, mp in enumerate(base_region_map[i]):
                bd = boundary_dict[i][j]
                if mp is None:
                    assert bd == 1, f"region[{i}] side[{j}] is not covered!"
                else:
                    assert bd == 0, f"region[{i}] side[{j}] is covered but it is not on boundary!"

    def _merge_boundary_dict(self, boundary_region_maps):
        """"""
        base_regions = self.regions._base_regions
        base_region_map = base_regions.map
        remaining_region_sides = {}

        for i in base_region_map:

            remaining_region_sides[i] = list()

            for j, mp in enumerate(base_region_map[i]):

                bds = list()
                for brm in boundary_region_maps:
                    if i not in brm:
                        bds.append(0)
                    else:
                        bds.append(
                            brm[i][j]
                        )

                count_bds = bds.count(1)
                assert count_bds <= 1, f"region boundary [{i}][{j}] appears more than once in boundary sections."

                if mp is None:
                    if 1 in bds:
                        assert count_bds == 1, f"must be"
                        remaining_region_sides[i].append(0)
                    else:
                        assert count_bds == 0, f"must be"
                        remaining_region_sides[i].append(1)
                else:
                    assert count_bds == 0, f"must be"
                    remaining_region_sides[i].append(0)

        return remaining_region_sides

    def __repr__(self):
        super_repr = super().__repr__().split('object')[1]
        return f"<{self.__class__.__name__} " + self._abstract._sym_repr + super_repr

    @property
    def abstract(self):
        return self._abstract

    @property
    def ndim(self):
        """The dimensions of this manifold (Not the embedding space dimensions!)"""
        return self._abstract.ndim

    @property
    def esd(self):
        """embedding space dimensions."""
        return get_embedding_space_dim()

    @property
    def m(self):
        return self.esd

    @property
    def n(self):
        return self.ndim

    @property
    def ct(self):
        if self._ct is None:
            self._ct = MsePyManifoldsCoordinateTransformation(self)
        return self._ct

    @property
    def regions(self):
        """"""
        assert self._regions is not None, f"regions of {self} is not configured, config it through `msepy.config`"
        return self._regions

    @property
    def visualize(self):
        if self._visualize is None:
            self._visualize = MsePyManifoldVisualize(self)
        return self._visualize
