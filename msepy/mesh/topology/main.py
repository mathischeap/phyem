# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 1:46 PM on 5/2/2023
"""

import sys

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen
import numpy as np


class MsePyMeshTopology(Frozen):
    """"""

    def __init__(self, mesh):
        """"""
        self._mesh = mesh
        self._corner_numbering = None
        self._corner_attachment = None
        self._freeze()

    @property
    def corner_numbering(self):
        if self._corner_numbering is None:
            self._generate_numbering_gathering_corners()
        return self._corner_numbering

    @property
    def corners_attachment(self):
        if self._corner_attachment is None:
            self._generate_numbering_gathering_corners()
        return self._corner_attachment

    def _generate_numbering_gathering_corners(self):
        """
        Here we try to generate the gathering matrix and attachment info of
        element corners.

        The idea is:
            (1): We go through all elements.
            (2): We go through all corners.
            (3): If corner numbered, we skip it, else we find all elements
                sharing the corner and number it.

        """
        n = self._mesh.n

        if n == 2:  # mesh on a 2-manifold
            c_i2n = {0: 'UL', 1: 'DL', 2: 'UR', 3: 'DR'}
            c_n2i = {'UL': 0, 'DL': 1, 'UR': 2, 'DR': 3,
                     'LU': 0, 'LD': 1, 'RU': 2, 'RD': 3}
            gathering = -np.ones((self._mesh.elements._num, 4))
            current = 0
            self._corner_attachment = {}
            for i in range(self._mesh.elements._num):
                for k in range(4):
                    if gathering[i, k] == -1:  # not numbered yet
                        # Now we first find the elements sharing this corner
                        corner_name = c_i2n[k]

                        self._corner_attachment[current] = self._find_the_elements_sharing_corner(
                            i, corner_name)

                        for item in self._corner_attachment[current]:
                            # now, we number other corners attached to this corner
                            try:
                                element, cn = item.split('-')
                                gathering[int(element), c_n2i[cn]] = current
                            except ValueError:
                                pass

                        assert gathering[i, k] != -1, " <Geometry> "  # this corner has to be numbered now.

                        current += 1
                    else:  # it is numbered, we skip it.
                        pass
            self._corner_numbering = gathering.astype(int)
        else:
            raise NotImplementedError()

    def _find_the_elements_sharing_corner(self, i, corner_name):
        """
        Find all elements and domain boundaries that contain the corner named
        `corner_name` of ith element.

        Parameters
        ----------
        i : natural_number
            ith mesh element
        corner_name : str
            The corner name

        Returns
        -------
        output : tuple
            A tuple of all elements and domain boundary that contains this corner.

        """

        n = self._mesh.n
        if n == 2:
            seek_sequence_1 = {'UL': '[L=UR]->[U=DR]->[R=DL]',
                               'DL': '[L=DR]->[D=UR]->[R=UL]',
                               'UR': '[U=DR]->[R=DL]->[D=UL]',
                               'DR': '[R=DL]->[D=UL]->[L=UR]'}[corner_name]
            seek_sequence_2 = {'UL': '[U=DL]->[L=DR]->[D=UR]',
                               'DL': '[D=UL]->[L=UR]->[U=DR]',
                               'UR': '[R=UL]->[U=DL]->[L=DR]',
                               'DR': '[D=UR]->[R=UL]->[U=DL]'}[corner_name]
            e_n2i = {'U': 0, 'D': 1, 'L': 2, 'R': 3}
            em = self._mesh.elements.map

            output = (str(i)+'-'+corner_name,)
            io = i  # save i original
            for ss in (seek_sequence_1, seek_sequence_2):
                i = io
                IJK = ss.split('->')
                for ijk in IJK:
                    direction, corner = ijk[1:-1].split('=')
                    if em[i][e_n2i[direction]] == -1:  # boundary
                        break
                    else:
                        i = em[i][e_n2i[direction]]
                        if str(i)+'-'+corner not in output:
                            output += (str(i)+'-'+corner,)
            return output
        else:
            raise NotImplementedError()


if __name__ == '__main__':
    # python 
    pass
