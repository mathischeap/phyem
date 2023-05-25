# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
created at: 3/31/2023 2:29 PM
"""
from tools.frozen import Frozen


class MseManifoldRegions(Frozen):
    """"""

    def __init__(self, mf):
        """"""
        # no need to do any check. It should be done already! We only access this class through `config`
        self._mf = mf
        self._regions = dict()
        self._map = None  # normally, only regions of the highest dimensional manifold has a region map.
        self._is_structured_regions = None
        self._base_regions = None
        self._map_type = -1
        self._freeze()

    @property
    def map(self):
        """
        - It has no _base_regions, map_type=0
        {    #x-  #x+, #y-, ...
            0: [int, int, None, ...],
            1: [int, None, int, ...],
            ...
        }
        this type of region map means it has a structured distribution of regions
        and int means this interface is in-between region, None means it is at manifold boundary.

        - It has no _base_regions: (map_type=1)
        {
            0: [0, 1, 0, ...],
            1: [...],
            ...,
        }
        This map indicate it covers these faces of the base regions.
        """
        return self._map  # ***

    def _is_periodic(self):
        """If there is no None in `self.map`, then it represents a periodic manifold."""
        if self._map_type == 0:
            is_periodic = True
            for i in self.map:
                mp = self.map[i]
                if None in mp:
                    is_periodic = False
                    break
            return is_periodic
        else:
            raise NotImplementedError(f"not implemented for map_type={self._map_type}.")

    def _check_map(self):
        """check region map."""
        map_type = self._map_type
        region_map = self.map
        assert region_map is not None, f"I have no map."
        if map_type == 0:  # the first type; indicating the neighbours.
            for i in self:
                map_i = region_map[i]
                for j, mp in enumerate(map_i):
                    if mp is None:
                        pass
                    else:
                        m = j // 2  # axis index, x -> y -> z ...
                        n = j % 2  # side index, 0: -, 1: +

                        map_neighbor = region_map[mp]
                        if n == 0:
                            _n = 1
                        else:
                            _n = 0

                        _j = m * 2 + _n

                        assert map_neighbor[_j] == i, \
                            f"region maps illegal; map[{i}][{j}] refers to region #{mp}, " \
                            f"but map[{mp}][{_j}] does not refer to region #{i}."
        elif map_type == 1:  # region boundary type regions
            pass
        else:
            raise NotImplementedError(f"cannot check for map_type={self._map_type}.")

    def is_structured(self):
        """Return True if we have a structured region map; `map_type=0` in `_check_map`;
        map is a 2d-array of integers and None.

        Else, return False.
        """
        if self._is_structured_regions is None:   # map_type = 0, see `_check_map`.
            if self._map_type == 0:
                self._is_structured_regions = True
            else:
                self._is_structured_regions = False
        return self._is_structured_regions

    def __repr__(self):
        """"""
        super_repr = super().__repr__().split('object')[1]
        return f"<Regions of " + self._mf.__repr__() + super_repr

    def __iter__(self):
        """go through all region index. Remember, a region does not have a name."""
        for ri in range(len(self)):  # do this because we make sure we start with region #0.
            yield ri

    def __getitem__(self, ri):
        """Retrieve a region through its index."""
        return self._regions[ri]

    def __len__(self):
        """How many regions I have?"""
        return len(self._regions)

    def __contains__(self, i):
        """check if `i` is a valid region index."""
        return i in self._regions

    @property
    def m(self):
        return self._mf.m

    @property
    def n(self):
        return self._mf.n
