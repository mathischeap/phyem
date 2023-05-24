# -*- coding: utf-8 -*-

from tools.frozen import Frozen
import numpy as np


class MsePyStandardRegionCoordinateTransformation(Frozen):
    """"""

    def __init__(self, mapping, Jacobian_matrix, mtype):

        self._mapping = mapping
        if Jacobian_matrix is None:
            raise NotImplementedError(f"Implement a numerical Jacobian_matrix")
        else:
            self._Jacobian_matrix = Jacobian_matrix
        self.mtype = mtype  # if it is None, we will set a unique one.
        self._freeze()

    @property
    def mapping(self):
        """"""
        return self._mapping

    @property
    def Jacobian_matrix(self):
        """"""
        return self._Jacobian_matrix

    @property
    def mtype(self):
        """"""
        return self._mtype

    @mtype.setter
    def mtype(self, mtp):
        """"""
        if mtp is None:
            indicator = 'Unique'
            parameters = None
        else:
            assert isinstance(mtp, dict), f"mtype must be a dict."
            indicator = mtp['indicator']
            parameters = mtp['parameters']
        mtp = _MsePyRegionMtype(indicator, parameters)
        self._mtype = mtp


class _MsePyRegionMtype(Frozen):
    """"""
    def __init__(self, indicator, parameters):
        assert indicator in (
            'Unique',  # `parameters` is a the unique id.
            'Linear',  # regular box in Cartesian system.
                       # `parameters` is a list of region length along each axis.
        ), f"indicator = {indicator} is illegal."
        if parameters is None:
            parameters = id(self)
        else:
            pass
        self._indicator = indicator
        self._parameters = parameters
        self._signature = self._indicator + ":" + str(self._parameters)
        self._freeze()

    @property
    def signature(self):
        """"""
        return self._signature

    def __eq__(self, other):
        """"""
        if other.__class__ is not _MsePyRegionMtype:
            return False
        else:
            return self.signature == other.signature

    def _distribute_to_element(self, layout, element_numbering):
        """"""
        element_mtype_dict = dict()

        if self._indicator == 'Unique':
            raise Exception(f"Unique region cannot distribute metric type to elements.")

        elif self._indicator == 'Linear':
            # parameters are for example: ['x1.33333', 'y1.666666', ...]
            assert len(layout) == len(self._parameters), f"layout or parameters dimensions wrong."
            assert len(layout) == element_numbering.ndim, f"layout or element_numbering dimensions wrong."

            LayOut = [np.round(_, 5) for _ in layout]
            parameters = self._parameters
            axis = [_[0] for _ in parameters]
            para = [float(_[1:]) for _ in parameters]

            if len(LayOut) == 1:
                for i in range(len(LayOut[0])):
                    Li = LayOut[0][i]
                    pi = para[0] * Li
                    pi = round(pi, 5)
                    key = 'Linear:' + axis[0] + str(pi)

                    element_number = element_numbering[i, ]
                    if key in element_mtype_dict:
                        element_mtype_dict[key].append(element_number)
                    else:
                        element_mtype_dict[key] = [element_number, ]

            elif len(LayOut) == 2:
                for j in range(len(LayOut[1])):
                    for i in range(len(LayOut[0])):

                        Li, Lj = LayOut[0][i], LayOut[1][j]
                        pi, pj = para[0] * Li, para[1] * Lj
                        pi = round(pi, 5)
                        pj = round(pj, 5)
                        key = 'Linear:' + axis[0] + str(pi) + axis[1] + str(pj)

                        element_number = element_numbering[i, j]
                        if key in element_mtype_dict:
                            element_mtype_dict[key].append(element_number)
                        else:
                            element_mtype_dict[key] = [element_number, ]

            elif len(LayOut) == 3:
                for k in range(len(LayOut[2])):
                    for j in range(len(LayOut[1])):
                        for i in range(len(LayOut[0])):

                            Li, Lj, Lk = LayOut[0][i], LayOut[1][j], LayOut[2][k]
                            pi, pj, pk = para[0] * Li, para[1] * Lj, para[2] * Lk
                            pi = round(pi, 5)
                            pj = round(pj, 5)
                            pk = round(pk, 5)
                            key = 'Linear:' + axis[0] + str(pi) + axis[1] + str(pj) + axis[2] + str(pk)

                            element_number = element_numbering[i, j, k]
                            if key in element_mtype_dict:
                                element_mtype_dict[key].append(element_number)
                            else:
                                element_mtype_dict[key] = [element_number, ]

            else:
                raise NotImplementedError()

        else:
            raise NotImplementedError(f"Not implemented for indicator = {self._indicator}")

        return element_mtype_dict
