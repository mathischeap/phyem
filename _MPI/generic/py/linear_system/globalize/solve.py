# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen

from _MPI.generic.py.matrix.globalize.static import MPI_PY_Globalize_Static_Matrix
from _MPI.generic.py.vector.globalize.static import MPI_PY_Globalize_Static_Vector

from _MPI.generic.py.linear_system.globalize.solvers._py_mpi import _PY_MPI_solvers
from _MPI.generic.py.linear_system.globalize.solvers._scipy import _PackageScipy


class MPI_PY_Solve(Frozen):
    """"""
    def __init__(self, A, b):
        """"""
        self._A = A   # MPI_PY_Globalize_Static_Matrix
        self._b = b   # MPI_PY_Globalize_Static_Vector
        assert A.__class__ is MPI_PY_Globalize_Static_Matrix, f"A needs to be a {MPI_PY_Globalize_Static_Matrix}"
        assert b.__class__ is MPI_PY_Globalize_Static_Vector, f"b needs to be a {MPI_PY_Globalize_Static_Vector}"

        self._package = 'scipy'
        self._scheme = 'spsolve'

        # implemented packages
        self._package_scipy = _PackageScipy()
        self._package_py_mpi = _PY_MPI_solvers()

        self._x0 = None
        self._freeze()

    @property
    def A(self):
        return self._A

    @property
    def b(self):
        return self._b

    @property
    def package(self):
        return self._package

    @package.setter
    def package(self, package):
        self._package = package

    @property
    def scheme(self):
        return self._scheme

    @scheme.setter
    def scheme(self, scheme):
        self._scheme = scheme

    @property
    def x0(self):
        """The initial guess for iterative solver."""
        return self._x0

    @x0.setter
    def x0(self, args):
        """"""
        if args == 0:  # make a full zero initial guess

            x0 = np.zeros(self.b.shape)

        elif all([hasattr(f, 'cochain') for f in args]):   # providing forms
            mesh = None
            for f in args:
                f_mesh = f.mesh
                if mesh is None:
                    mesh = f_mesh
                else:
                    assert mesh is f_mesh, f"mesh must be consistent!"

            if hasattr(mesh, 'generic'):
                mesh = mesh.generic
            else:
                pass

            # use the newest cochains.
            cochain = dict()
            for local_element_index in mesh:
                cochain[local_element_index] = list()
            for f in args:
                newest_time = f.cochain.newest
                if newest_time is None:  # no newest cochain at all.
                    # then use 0-cochain
                    gm = f.cochain.gathering_matrix
                    local_cochain = dict()
                    for local_element_index in gm:
                        local_cochain[local_element_index] = np.zeros(
                            gm.num_local_dofs(local_element_index)
                        )
                else:
                    local_cochain = f.cochain[newest_time].local
                for local_element_index in mesh:
                    cochain[local_element_index].extend(local_cochain[local_element_index])

            from _MPI.generic.py.gathering_matrix import MPI_PyGM
            chain_gm = MPI_PyGM(*[f.cochain.gathering_matrix for f in args])
            x0 = chain_gm.assemble(cochain, mode='replace', globalize=True)

        else:
            raise NotImplementedError()

        assert isinstance(x0, np.ndarray), f"x0 must be a ndarray."
        assert x0.shape == self.b.shape, f"x0 shape wrong!"
        assert x0.shape == (self.A.shape[1], ), f"x0 shape wrong!"
        self._x0 = x0

    def __call__(self, **kwargs):
        """

        Parameters
        ----------
        kwargs :
            The parameters to be passed to the solver (except A, b, x0. They will be passed automatically
            using properties of self).

        Returns
        -------

        """
        assert hasattr(self, f"_package_{self.package}"), f"I have no solver package: {self.package}."
        _package = getattr(self, f"_package_{self.package}")
        assert hasattr(_package, self.scheme), f"package {self.package} has no scheme: {self.scheme}"
        solver = getattr(_package, self.scheme)

        if self.x0 is None:  # do this since direct solver does not need x0.
            x, message, info = solver(self.A, self.b, **kwargs)
        else:
            x, message, info = solver(self.A, self.b, self.x0, **kwargs)

        return x, message, info
