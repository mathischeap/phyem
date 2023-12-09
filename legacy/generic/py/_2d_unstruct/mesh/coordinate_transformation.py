# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class Py2CoordinateTransformation(Frozen):
    """"""

    def __init__(self, mesh):
        """"""
        self._mesh = mesh
        self._freeze()

    def _parse_element_range(self, element_range):
        """"""
        if element_range is None:
            element_range = self._mesh._elements_dict
        else:
            if isinstance(element_range, int):
                element_range = [element_range, ]
            else:
                pass
            assert isinstance(element_range, (list, tuple)), \
                f'put element indices in a list or tuple.'
            for index in element_range:
                assert index in self._mesh, f"element #{index} is not valid."
        return element_range

    def mapping(self, xi, et, element_range=None):
        """"""
        element_range = self._parse_element_range(element_range)
        xy = dict()
        for index in element_range:
            xy[index] = self._mesh[index].ct.mapping(xi, et)
        return xy

    def Jacobian_matrix(self, xi, et, element_range=None):
        """"""
        element_range = self._parse_element_range(element_range)
        JM = dict()
        cache = dict()
        for index in element_range:
            element = self._mesh[index]
            metric_signature = element.metric_signature
            if metric_signature in cache:
                JM[index] = cache[metric_signature]
            else:
                m = self._mesh[index].ct.Jacobian_matrix(xi, et)
                JM[index] = m
                cache[metric_signature] = m
        return JM

    def Jacobian(self, xi, et, element_range=None):
        """"""
        element_range = self._parse_element_range(element_range)
        J = dict()
        cache = dict()
        for index in element_range:
            element = self._mesh[index]
            metric_signature = element.metric_signature
            if metric_signature in cache:
                J[index] = cache[metric_signature]
            else:
                m = self._mesh[index].ct.Jacobian(xi, et)
                J[index] = m
                cache[metric_signature] = m
        return J

    def inverse_Jacobian_matrix(self, xi, et, element_range=None):
        """"""
        element_range = self._parse_element_range(element_range)
        iJm = dict()
        cache = dict()
        for index in element_range:
            element = self._mesh[index]
            metric_signature = element.metric_signature
            if metric_signature in cache:
                iJm[index] = cache[metric_signature]
            else:
                m = self._mesh[index].ct.inverse_Jacobian_matrix(xi, et)
                iJm[index] = m
                cache[metric_signature] = m
        return iJm

    def inverse_Jacobian(self, xi, et, element_range=None):
        """"""
        element_range = self._parse_element_range(element_range)
        iJ = dict()
        cache = dict()
        for index in element_range:
            element = self._mesh[index]
            metric_signature = element.metric_signature
            if metric_signature in cache:
                iJ[index] = cache[metric_signature]
            else:
                m = self._mesh[index].ct.inverse_Jacobian(xi, et)
                iJ[index] = m
                cache[metric_signature] = m
        return iJ

    def inverse_metric_matrix(self, xi, et, element_range=None):
        """"""
        element_range = self._parse_element_range(element_range)
        imm = dict()
        cache = dict()
        for index in element_range:
            element = self._mesh[index]
            metric_signature = element.metric_signature
            if metric_signature in cache:
                imm[index] = cache[metric_signature]
            else:
                m = self._mesh[index].ct.inverse_metric_matrix(xi, et)
                imm[index] = m
                cache[metric_signature] = m
        return imm

    def metric_matrix(self, xi, et, element_range=None):
        """"""
        element_range = self._parse_element_range(element_range)
        mm = dict()
        cache = dict()
        for index in element_range:
            element = self._mesh[index]
            metric_signature = element.metric_signature
            if metric_signature in cache:
                mm[index] = cache[metric_signature]
            else:
                m = self._mesh[index].ct.metric_matrix(xi, et)
                mm[index] = m
                cache[metric_signature] = m
        return mm
