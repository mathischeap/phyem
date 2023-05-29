# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 1:46 PM on 5/2/2023
"""
from tools.frozen import Frozen
import numpy as np


class MsePyMeshTopology(Frozen):
    """"""

    def __init__(self, mesh):
        """"""
        self._mesh = mesh
        self._corner_numbering = None
        self._corner_attachment = None
        self._edge_numbering = None
        self._edge_attachment = None
        self._side_numbering = None
        self._side_attachment = None
        self._freeze()

    @property
    def corner_numbering(self):
        if self._corner_numbering is None:
            self._generate_numbering_gathering_corners()
        return self._corner_numbering

    @property
    def corner_attachment(self):
        if self._corner_attachment is None:
            self._generate_numbering_gathering_corners()
        return self._corner_attachment

    @property
    def edge_numbering(self):
        """ The gathering numbering of all element edges. """
        if self._edge_numbering is None:
            self._generate_numbering_gathering_edges_()
        return self._edge_numbering

    @property
    def edge_attachment(self):
        """
        Here we store which edges are shared by which elements.

        Returns
        -------
        self._attachment_edge_ : dict
            A dict whose keys are the numbering of element edges. For example:
                {
                    0: {'0-WB', },
                    1: {'0-EB', '2-WB', },
                    2: {'0-WF', '4-BW', },
                    3: {'0-EF', '2-WF', '4-EB', '6-BW'},
                    4: {'1-WB', },
                    ...
                }
            means the edge numbered 0 is the West-Back edge of element 0.

        """
        if self._edge_attachment is None:
            self._generate_numbering_gathering_edges_()
        return self._edge_attachment

    @property
    def side_numbering(self):
        """ The gathering numbering of all element sides. """
        if self._side_numbering is None:
            self._generate_numbering_gathering_sides_()
        return self._side_numbering

    @property
    def side_attachment(self):
        """ """
        if self._side_attachment is None:
            self._generate_numbering_gathering_sides_()
        return self._side_attachment

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
        elif n == 3:
            c_i2n = {0: 'NWB', 1: 'SWB', 2: 'NEB', 3: 'SEB', 4: 'NWF', 5: 'SWF', 6: "NEF", 7: 'SEF'}
            c_n2i = {
                'NWB': 0, 'SWB': 1, 'NEB': 2, 'SEB': 3, 'NWF': 4, 'SWF': 5, "NEF": 6, 'SEF': 7,
                'NBW': 0, 'SBW': 1, 'NBE': 2, 'SBE': 3, 'NFW': 4, 'SFW': 5, "NFE": 6, 'SFE': 7,
                'WNB': 0, 'WSB': 1, 'ENB': 2, 'ESB': 3, 'WNF': 4, 'WSF': 5, "ENF": 6, 'ESF': 7,
                'WBN': 0, 'WBS': 1, 'EBN': 2, 'EBS': 3, 'WFN': 4, 'WFS': 5, "EFN": 6, 'EFS': 7,
                'BNW': 0, 'BSW': 1, 'BNE': 2, 'BSE': 3, 'FNW': 4, 'FSW': 5, "FNE": 6, 'FSE': 7,
                'BWN': 0, 'BWS': 1, 'BEN': 2, 'BES': 3, 'FWN': 4, 'FWS': 5, "FEN": 6, 'FES': 7
            }

            gathering = -np.ones((self._mesh.elements._num, 8))
            current_num = 0
            self._corner_attachment = {}
            for i in range(self._mesh.elements._num):
                for k in range(8):
                    if gathering[i, k] == -1:  # not numbered yet
                        # Now we first find the elements sharing this corner
                        corner_name = c_i2n[k]

                        _attachment_corners_1_ = \
                            self._find_the_elements_sharing_corner_edge(i, corner_name[0:2])
                        _attachment_corners_2_ = \
                            self._find_the_elements_sharing_corner_edge(i, corner_name[1:3])
                        _attachment_corners_3_ = \
                            self._find_the_elements_sharing_corner_edge(i, corner_name[0:3:2])
                        self._corner_attachment[current_num] = self._group_edges_into_corners(
                            i,
                            _attachment_corners_1_,
                            _attachment_corners_2_,
                            _attachment_corners_3_
                        )
                        for item in self._corner_attachment[current_num]:
                            # now, we number this group corners
                            try:
                                element, cn = item.split('-')
                                gathering[int(element), c_n2i[cn]] = current_num
                            except ValueError:
                                pass
                        assert gathering[i, k] != -1, " <Geometry3D> "  # this edge has to be numbered now.
                        current_num += 1
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
            seek_sequence_1 = {
                'UL': '[L=UR]->[U=DR]->[R=DL]',
                'DL': '[L=DR]->[D=UR]->[R=UL]',
                'UR': '[U=DR]->[R=DL]->[D=UL]',
                'DR': '[R=DL]->[D=UL]->[L=UR]',
            }[corner_name]

            seek_sequence_2 = {
                'UL': '[U=DL]->[L=DR]->[D=UR]',
                'DL': '[D=UL]->[L=UR]->[U=DR]',
                'UR': '[R=UL]->[U=DL]->[L=DR]',
                'DR': '[D=UR]->[R=UL]->[U=DL]',
            }[corner_name]

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

    def _find_the_elements_sharing_corner_edge(self, i, edge_name):
        """
        Find all elements and domain boundaries that contain the edge named
        `edge_name` of ith element.

        Parameters
        ----------
        i : natural_number
            ith element
        edge_name : str
            The edge name

        Returns
        -------
        output : tuple
            A tuple of all elements and domain boundary that contains this edge.

        """
        s_n2i = {'N': 0, 'S': 1, 'W': 2, 'E': 3, 'B': 4, 'F': 5}
        s_pair = {'N': 'S', 'S': 'N', 'W': 'E', 'E': 'W', 'B': 'F', 'F': 'B'}

        _tuple_ = (str(i)+'-'+edge_name,)

        s1, s2 = edge_name
        # Now we go along this two direction to reach element i or boundary
        location = self._mesh.elements.map[i][s_n2i[s1]]

        num_while = 0
        while location != -1 and location != i:
            num_while += 1
            _tuple_ += (str(location)+'-'+s_pair[s1]+s2,)
            s1, s2 = s2, s_pair[s1]
            location = self._mesh.elements.map[location][s_n2i[s1]]
            if num_while > 3:
                raise Exception(' <Mesh> <Geometry> : mesh is wrong')
        if location == -1:  # if the while loop stops because reaching the boundary.
            s2, s1 = edge_name  # then perform the same while loop to find every thing but from another direction.
            location = self._mesh.elements.map[i][s_n2i[s1]]
            num_while = 0
            while location != -1 and location != i:
                num_while += 1
                _tuple_ += (str(location)+'-'+s_pair[s1]+s2,)
                s1, s2 = s2, s_pair[s1]
                location = self._mesh.elements.map[location][s_n2i[s1]]
                if num_while > 3:
                    raise Exception(' <Mesh> <Geometry> : mesh is wrong')
            if location == -1:  # the while loop stops because reaching the boundary again
                pass

        _edge_index_to_name_ = {
            0: 'WB', 1: 'EB', 2: 'WF', 3: 'EF', 4: 'NB', 5: 'SB',
            6: 'NF', 7: 'SF', 8: 'NW', 9: 'SW', 10: 'NE', 11: 'SE'
        }
        _edge_name_to_index_ = {
            'WB': 0, 'EB': 1, 'WF': 2, 'EF': 3, 'NB': 4, 'SB': 5,
            'NF': 6, 'SF': 7, 'NW': 8, 'SW': 9, 'NE': 10, 'SE': 11,
            'BW': 0, 'BE': 1, 'FW': 2, 'FE': 3, 'BN': 4, 'BS': 5,
            'FN': 6, 'FS': 7, 'WN': 8, 'WS': 9, 'EN': 10, 'ES': 11
        }
        output = ()
        for item in _tuple_:
            element, edge_name = item.split('-')
            edge_name = _edge_index_to_name_[_edge_name_to_index_[edge_name]]
            output += (element + '-' + edge_name,)

        return output

    def _group_edges_into_corners(self, i, *args):
        """
        As we use the method ___find_the_elements_sharing_corner_edge___ to find the
        nearby elements, the format in the attachment is as {'7-NE', '7-NF',
        ...}, we need to group this item into '7-NEF'.

        """
        _dict_ = {}
        for ac in args:
            for item in ac:
                ith_element = int(item.split('-')[0])
                edge = item.split('-')[1]
                assert 0 <= int(ith_element) < self._mesh.elements._num, " <Geometry3D> "
                if ith_element not in _dict_:
                    _dict_[ith_element] = []
                _dict_[ith_element].append(edge)

        for key in _dict_:
            if isinstance(key, int):
                _dict_[key] = ''.join(list(set(''.join(_dict_[key]))))
            else:
                raise Exception()

        for key in _dict_:
            if isinstance(key, int):
                if len(_dict_[key]) == 2:
                    # there are some element or edges which are not saw
                    # we have to give this method more information and return
                    # the new output
                    ARGS = []
                    for I_ in _dict_:
                        if I_ != i:
                            if len(_dict_[I_]) == 3:
                                ARGS.append(self._find_the_elements_sharing_corner_edge(I_, _dict_[I_][0:2]))
                                ARGS.append(self._find_the_elements_sharing_corner_edge(I_, _dict_[I_][1:3]))
                                ARGS.append(self._find_the_elements_sharing_corner_edge(I_, _dict_[I_][0:3:2]))
                    return self._group_edges_into_corners(i, *args, *ARGS)

                elif len(_dict_[key]) == 3:
                    pass
                else:
                    raise Exception

            else:
                raise Exception

        _corner_index_to_name_ = {
            0: 'NWB', 1: 'SWB', 2: 'NEB', 3: 'SEB', 4: 'NWF', 5: 'SWF', 6: "NEF", 7: 'SEF'
        }
        _corner_name_to_index_ = {
            'NWB': 0, 'SWB': 1, 'NEB': 2, 'SEB': 3, 'NWF': 4, 'SWF': 5, "NEF": 6, 'SEF': 7,
            'NBW': 0, 'SBW': 1, 'NBE': 2, 'SBE': 3, 'NFW': 4, 'SFW': 5, "NFE": 6, 'SFE': 7,
            'WNB': 0, 'WSB': 1, 'ENB': 2, 'ESB': 3, 'WNF': 4, 'WSF': 5, "ENF": 6, 'ESF': 7,
            'WBN': 0, 'WBS': 1, 'EBN': 2, 'EBS': 3, 'WFN': 4, 'WFS': 5, "EFN": 6, 'EFS': 7,
            'BNW': 0, 'BSW': 1, 'BNE': 2, 'BSE': 3, 'FNW': 4, 'FSW': 5, "FNE": 6, 'FSE': 7,
            'BWN': 0, 'BWS': 1, 'BEN': 2, 'BES': 3, 'FWN': 4, 'FWS': 5, "FEN": 6, 'FES': 7
        }
        output = set()
        for key in _dict_:
            if isinstance(key, int):
                corner_name = _corner_index_to_name_[_corner_name_to_index_[_dict_[key]]]
                output.update({str(key)+'-'+corner_name, })
            else:
                raise Exception
        output = tuple(output)  # we do this to avoid uncertainty during using `set`.
        return output

    def _generate_numbering_gathering_edges_(self):
        """
        Here we generate the gathering numbering of element edges. In the
        meantime, we will generate the edge attachment info.

        The idea is,
            (1): We go through all elements.
            (2): In each element, we go through all edges
            (3): If numbered, skip. If not, find the all 3 (at most) other
                 elements that is sharing this edge by using method:
                 ___find_the_elements_sharing_edge___.
            (4): Number the edge in all 4 elements.

        """
        if self._mesh.n == 3:
            e_i2n = {
                0: 'WB', 1: 'EB', 2: 'WF', 3: 'EF', 4: 'NB', 5: 'SB',
                6: 'NF', 7: 'SF', 8: 'NW', 9: 'SW', 10: 'NE', 11: 'SE'
            }
            e_n2i = {
                'WB': 0, 'EB': 1, 'WF': 2, 'EF': 3, 'NB': 4, 'SB': 5,
                'NF': 6, 'SF': 7, 'NW': 8, 'SW': 9, 'NE': 10, 'SE': 11,
                'BW': 0, 'BE': 1, 'FW': 2, 'FE': 3, 'BN': 4, 'BS': 5,
                'FN': 6, 'FS': 7, 'WN': 8, 'WS': 9, 'EN': 10, 'ES': 11
            }

            gathering = -np.ones((self._mesh.elements._num, 12))
            correct_num = 0
            self._edge_attachment = {}
            for i in range(self._mesh.elements._num):
                for k in range(12):
                    if gathering[i, k] == -1:  # not numbered yet
                        # Now we first find the elements sharing this edge
                        edge_name = e_i2n[k]
                        self._edge_attachment[correct_num] = \
                            self._find_the_elements_sharing_corner_edge(i, edge_name)
                        for item in self._edge_attachment[correct_num]:
                            # now, we number this group edges

                            element, en = item.split('-')
                            gathering[int(element), e_n2i[en]] = correct_num

                        assert gathering[i, k] != -1, " <Geometry3D> "  # this edge has to be numbered now.
                        correct_num += 1
                    else:  # it is numbered, we skip it.
                        pass
            for i in self._edge_attachment:
                self._edge_attachment[i] = tuple(dict.fromkeys(self._edge_attachment[i]))
            self._edge_numbering = gathering.astype(int)
        else:
            raise Exception()

    def _generate_numbering_gathering_sides_(self):
        """ """
        if self._mesh.n == 3:
            s_n2i = {'N': 0, 'S': 1, 'W': 2, 'E': 3, 'B': 4, 'F': 5}
            s_i2n = {0: 'N', 1: 'S', 2: 'W', 3: 'E', 4: 'B', 5: 'F'}
            s_pair = {'N': 'S', 'S': 'N', 'W': 'E', 'E': 'W', 'B': 'F', 'F': 'B'}

            gathering = -np.ones((self._mesh.elements._num, 6))
            correct_num = 0
            self._side_attachment = {}
            for i in range(self._mesh.elements._num):
                for k in range(6):
                    if gathering[i, k] == -1:  # not numbered yet
                        # Now we first find the element sharing this side
                        self_side_name = s_i2n[k]
                        what = self._mesh.elements.map[i][k]
                        if what == -1:
                            self._side_attachment[correct_num] = (str(i) + '-' + self_side_name, )
                        else:
                            other_side_name = s_pair[self_side_name]
                            self._side_attachment[correct_num] = (str(i) + '-' + self_side_name,
                                                                  str(what) + '-' + other_side_name)
                        for item in self._side_attachment[correct_num]:
                            # now, we number this group edges

                            element, sn = item.split('-')
                            gathering[int(element), s_n2i[sn]] = correct_num

                        assert gathering[i, k] != -1, " <Mesh> <Geometry> "  # this side has to be numbered now.
                        correct_num += 1
                    else:  # it is numbered, we skip it.
                        pass
            self._side_numbering = gathering.astype(int)

        else:
            raise Exception()
