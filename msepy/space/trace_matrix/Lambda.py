# -*- coding: utf-8 -*-
r"""
"""
from scipy.sparse import lil_matrix

from phyem.tools.frozen import Frozen


class MsePyTraceMatrixLambda(Frozen):
    """"""

    def __init__(self, space):
        """Store required info."""
        self._space = space
        self._mesh = space.mesh
        self._k = space.abstract.k
        self._n = space.abstract.n  # manifold dimensions
        self._m = space.abstract.m  # dimensions of the embedding space.
        self._orientation = space.abstract.orientation
        self._cache = dict()
        self._freeze()

    def __call__(self, degree):
        """Making the trace matrix for degree."""

        p = self._space[degree].p
        key = f"{p}"

        if key in self._cache:
            tM = self._cache[key]
        else:

            m = self._m
            n = self._n
            k = self._k

            if m == 2 and n == 2 and k == 1:  # for k == 0 and k == 1.
                method_name = f"_m{m}_n{n}_k{k}_{self._orientation}"
            else:
                method_name = f"_m{m}_n{n}_k{k}"
            tM = getattr(self, method_name)(degree)

            self._cache[key] = tM

        return tM

    def _m2_n2_k1_outer(self, degree):
        """"""
        from msepy.main import base
        meshes = base['meshes']
        boundary_sym = self._mesh.abstract.boundary()._sym_repr
        boundary_section = None
        for sym in meshes:
            if sym == boundary_sym:
                boundary_section = meshes[sym]
                break
            else:
                pass
        assert boundary_section is not None, f"must have found a boundary section."

        nWE, nNS = self._space[degree].p

        N_s__N_e = [_ for _ in range(0, nNS)]
        S_s__S_e = [_ for _ in range(nNS, 2 * nNS)]
        W_s__W_e = [_ for _ in range(2 * nNS, 2 * nNS + nWE)]
        E_s__E_e = [_ for _ in range(2 * nNS + nWE, 2 * nNS + 2 * nWE)]

        tM = dict()

        shape = (
            2 * (nWE + nNS),                    # num trace form dofs
            (nWE + 1) * nNS + nWE * (nNS + 1)   # num form dofs.
        )

        for i in range(self._mesh.elements._num):
            tM[i] = lil_matrix(shape)

        local_numbering = self._space.local_numbering(degree)

        faces = boundary_section.faces
        for i in faces:
            face = faces[i]
            m, n, element = face._m, face._n, face._element
            if m == 0 and n == 0:  # North
                tM[element][N_s__N_e, local_numbering[0][0, :]] = +1
            elif m == 0 and n == 1:  # South
                tM[element][S_s__S_e, local_numbering[0][-1, :]] = -1   # try it
            elif m == 1 and n == 0:  # West
                tM[element][W_s__W_e, local_numbering[1][:, 0]] = -1    # try it
            elif m == 1 and n == 1:  # East
                tM[element][E_s__E_e, local_numbering[1][:, -1]] = +1
            else:
                raise Exception()

        for i in tM:
            # noinspection PyUnresolvedReferences
            tM[i] = tM[i].tocsr()

        return tM
