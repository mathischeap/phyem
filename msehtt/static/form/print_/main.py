# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen


class MseHtt_Static_Form_Print(Frozen):
    r"""We can print some properties, attributes or something else of a static form a MseHtt implementation.

    This is usually for testing or checking or debugging purpose.
    """

    def __init__(self, f):
        r""""""
        self._f = f
        self._freeze()

    def mass_matrix(self, element_index):
        r"""

        Parameters
        ----------
        element_index :
            We will print the mass matrix of the element indexed `element_index`.

        Returns
        -------

        """

        mm = self._f.space.mass_matrix(self._f.degree)[0]
        if element_index in mm:
            print(mm[element_index].toarray(), flush=True)
        else:
            pass

    def incidence_matrix(self, element_index):
        r"""

        Parameters
        ----------
        element_index :
            We will print the incidence matrix of the element indexed `element_index`.

        Returns
        -------

        """
        im = self._f.space.incidence_matrix(self._f.degree)[0]
        if element_index in im:
            print(im[element_index].toarray(), flush=True)
        else:
            pass
