# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
_global_cache_k1_inner = dict()
_global_cache_k1_outer = dict()


class _MseHyPy2SpaceFindLocalDofs(Frozen):
    """"""

    def __init__(self, space):
        self._space = space
        self._cache = dict()
        self._freeze()

    def __call__(self, q_or_t, edge_index, p):
        """Find the local dofs on `m`-axis, `n`-side of the space of degree `degree`.

        So m = {0, 1, 2, ...}, n = {0, 1}.

        """
        key = (q_or_t, edge_index, p)
        if key in self._cache:
            return self._cache[key]
        else:
            pass

        px, py = p
        assert px == py, 'must be'

        space = self._space
        indicator = space.abstract.indicator
        _k = space.abstract.k
        _orientation = space.abstract.orientation

        if indicator == 'Lambda':
            if _k == 1 and _orientation == 'inner':
                edge_local_dofs = self._Lambda_k1_inner(q_or_t, edge_index, p)
            elif _k == 1 and _orientation == 'outer':
                edge_local_dofs = self._Lambda_k1_outer(q_or_t, edge_index, p)
            else:
                raise Exception()
        else:
            raise Exception()

        self._cache[key] = edge_local_dofs
        return self._cache[key]

    def _Lambda_k1_inner(self, q_or_t, edge_index, p):
        """"""
        key = q_or_t + str(edge_index) + str(p)
        if key in _global_cache_k1_inner:
            return _global_cache_k1_inner[key]
        else:
            pass

        px, py = p
        assert px == py, f"must be!"
        local_numbering = self._space.local_numbering.Lambda._k1_inner(p)
        if q_or_t == 't':
            assert edge_index in ('b', 0, 1), 'edge index wrong'
            local_numbering = local_numbering['t']
            dx, dy = local_numbering

            if edge_index == 'b':
                _ = dy[-1, :]
            elif edge_index == 0:
                _ = dx[:, 0]
            elif edge_index == 1:
                _ = dx[:, -1]
            else:
                raise Exception

        elif q_or_t == 'q':
            assert edge_index in range(4), 'edge index wrong'
            local_numbering = local_numbering['q']
            dx, dy = local_numbering

            if edge_index == 0:
                _ = dy[0, :]
            elif edge_index == 1:
                _ = dy[-1, :]
            elif edge_index == 2:
                _ = dx[:, 0]
            elif edge_index == 3:
                _ = dx[:, -1]
            else:
                raise Exception
        else:
            raise Exception

        _global_cache_k1_inner[key] = _
        return _

    def _Lambda_k1_outer(self, q_or_t, edge_index, p):
        """"""
        key = q_or_t + str(edge_index) + str(p)
        if key in _global_cache_k1_outer:
            return _global_cache_k1_outer[key]
        else:
            pass

        px, py = p
        assert px == py, f"must be!"
        local_numbering = self._space.local_numbering.Lambda._k1_outer(p)
        if q_or_t == 't':
            assert edge_index in ('b', 0, 1), 'edge index wrong'
            local_numbering = local_numbering['t']
            dy, dx = local_numbering

            if edge_index == 'b':
                _ = dy[-1, :]
            elif edge_index == 0:
                _ = dx[:, 0]
            elif edge_index == 1:
                _ = dx[:, -1]
            else:
                raise Exception

        elif q_or_t == 'q':
            assert edge_index in range(4), 'edge index wrong'
            local_numbering = local_numbering['q']
            dy, dx = local_numbering

            if edge_index == 0:
                _ = dy[0, :]
            elif edge_index == 1:
                _ = dy[-1, :]
            elif edge_index == 2:
                _ = dx[:, 0]
            elif edge_index == 3:
                _ = dx[:, -1]
            else:
                raise Exception

        else:
            raise Exception

        _global_cache_k1_outer[key] = _
        return _
