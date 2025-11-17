# -*- coding: utf-8 -*-
# noinspection PyUnresolvedReferences
r"""
Each space is an instance of a particular space class which inherits :class:`SpaceBase`.

    .. autoclass:: src.spaces.base.SpaceBase
        :members: mesh, manifold, m, n, make_form


"""
from time import time

from phyem.tools.frozen import Frozen
from phyem.src.config import get_embedding_space_dim
from phyem.src.form.main import Form
from phyem.src.spaces.finite import SpaceFiniteSetting
from phyem.tools.miscellaneous.random_ import string_digits


class SpaceBase(Frozen):
    """"""

    def __init__(self, mesh, orientation):
        """"""
        self._objective = None
        # when initializing, it has no objective instance. And when we generate an objective of a
        # abstract space, we store the last objective one with this attribute.

        self._mesh = mesh
        assert orientation in ('inner', 'outer', 'i', 'o', None, 'None'), \
            f"orientation={orientation} is wrong, must be one of ('inner', 'outer', 'i', 'o', None)."
        if orientation == 'i':
            orientation = 'inner'
        elif orientation == 'o':
            orientation = 'outer'
        elif orientation in (None, 'unknown', 'None'):
            orientation = 'unknown'
        else:
            pass
        self._orientation = orientation
        self._finite = None  # the finite setting

    @property
    def _pure_lin_repr(self):
        """"""
        raise NotImplementedError()

    @property
    def mesh(self):
        """The mesh I am built on."""
        return self._mesh

    @property
    def manifold(self):
        """The manifold I am built on."""
        return self.mesh.manifold

    @property
    def orientation(self):
        """My orientation."""
        return self._orientation

    @property
    def opposite_orientation(self):
        if self.orientation == 'inner':
            return 'outer'
        elif self.orientation == 'outer':
            return 'inner'
        elif self.orientation == 'unknown':
            return 'unknown'
        else:
            raise Exception()

    @property
    def n(self):
        """the dimensions of the mesh I am living in."""
        return self.mesh.ndim

    @property
    def m(self):
        """the dimensions of the space I am living in."""
        return get_embedding_space_dim()

    def make_form(
            self,
            sym_repr, lin_repr,
            dual_representation=False
    ):
        """Define a form which is an element of this space.

        Parameters
        ----------
        sym_repr : str
            The symbolic representation of the form.
        lin_repr : str
            The linguistic representation of the form.
        dual_representation : bool, optional
            Whether the output form uses dual representation? The default value is ``False``.

        Returns
        -------
        form : :class:`src.form.main.Form`
            The output form.

        """
        assert isinstance(sym_repr, str), f"symbolic representation must be a str."
        form = Form(
            self, sym_repr, lin_repr,
            True,  # is_root
        )
        if dual_representation:
            form.set_dual_representation(True)
        else:
            pass
        return form

    def make_random_form(self):
        """Usually used as intermediate forms."""
        sym = string_digits(8)
        t = time()
        random_sym_repr = sym+str(t)[-5:]
        random_lin_repr = str(hash(random_sym_repr))
        return self.make_form(random_sym_repr, random_lin_repr)

    def __eq__(self, other):
        """"""
        return self.__repr__() == other.__repr__()

    @staticmethod
    def _is_space():
        """A private signature/tag."""
        return True

    @property
    def finite(self):
        if self._finite is None:
            self._finite = SpaceFiniteSetting(self)
        return self._finite

    def d(self):
        """d (self)"""
        from phyem.src.spaces.operators import d
        return d(self)
