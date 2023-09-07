# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehy.py2.mesh.elements.main import MseHyPy2MeshElements
from msehy.py2.mesh.faces.main import MseHyPy2MeshFaces


class _MesHyPy2MeshGenerations(Frozen):
    """"""
    def __init__(self, mesh):
        """"""
        self._mesh = mesh
        self._pool = dict()
        self._max_generations = 2
        self._freeze()

    def sync_cache(self, cache_dict, generation, personal_key, data=None):
        """"""
        generation_data_to_be_cleaned = list()
        for cg in cache_dict:
            if cg not in self._pool:
                generation_data_to_be_cleaned.append(cg)
            else:
                pass
        for cg in generation_data_to_be_cleaned:
            del cache_dict[cg]

        if data is None:  # we are checking if the personal_key at `generation` is cached.
            if generation in cache_dict:
                if personal_key in cache_dict[generation]:
                    return True, cache_dict[generation][personal_key]
            return False, None
        else:   # we are doing the caching
            if len(self._pool) == 0:
                _ = self._mesh.current_representative

            if generation in self._pool:  # we will cache it

                if generation not in cache_dict:
                    cache_dict[generation] = dict()
                else:
                    pass
                cache_dict[generation][personal_key] = data  # cache it.

            else:  # do not cache it
                pass

    def _add(self, new_generation):
        """"""
        # delete extra generations
        if len(self._pool) >= int(1.5 * self._max_generations):  # * 1.5 for a margin.
            new_pool = dict()
            keys = list(self._pool.keys())
            keys.sort()
            new_keys = keys[:(self._max_generations-1)]
            for key in new_keys:
                new_pool[key] = self._pool[key]
            self._pool = new_pool
        else:
            pass

        if self._mesh._is_mesh():
            assert new_generation.__class__ is MseHyPy2MeshElements
        else:
            assert new_generation.__class__ is MseHyPy2MeshFaces

        g = new_generation.generation
        assert g not in self._pool, f"must be"
        if len(self._pool) > 0:
            assert g == max(self._pool.keys()) + 1, 'must be'
        else:
            assert len(self._pool) == 0 and g == 0, 'must be'

        self._pool[g] = new_generation

    def __getitem__(self, generation):
        """"""
        if len(self._pool) == 0:
            _ = self._mesh.current_representative
        assert generation in self._pool, f'generation [{generation}] is not cached.'
        return self._pool[generation]

    def __repr__(self):
        """repr"""
        return f"<Generation storage of {self._mesh}>"

    def __len__(self):
        """How many generations I am caching."""
        if len(self._pool) == 0:
            _ = self._mesh.current_representative
        return len(self._pool)

    def __contains__(self, generation):
        """If a generation is cached"""
        if len(self) == 0:
            _ = self._mesh.current_representative
        return generation in self._pool
