# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen


class CT(Frozen):
    """"""

    def __init__(self, elements):
        self._elements = elements
        self._freeze()

    def __repr__(self):
        """"""
        return rf"<CT of {self._elements}>"

    def ___parse_element_range_CT___(self, element_range):
        """"""
        if element_range is None:
            element_range = self._elements._elements_dict
        else:
            if isinstance(element_range, (int, str)):
                element_range = [element_range, ]
            else:
                pass
            for index in element_range:
                assert index in self._elements, f"element #{index} is not local."
        return element_range

    def mapping(self, xi, et, element_range=None):
        """"""
        element_range = self.___parse_element_range_CT___(element_range)
        mp = dict()
        for index in element_range:
            mp[index] = self._elements[index].ct.mapping(xi, et)
        return mp

    def Jacobian_matrix(self, xi, et, element_range=None):
        """"""
        element_range = self.___parse_element_range_CT___(element_range)
        JM = dict()
        cache = dict()
        for index in element_range:
            element = self._elements[index]
            metric_signature = element.metric_signature
            if metric_signature in cache:
                JM[index] = cache[metric_signature]
            else:
                m = element.ct.Jacobian_matrix(xi, et)
                JM[index] = m
                cache[metric_signature] = m
        return JM

    def Jacobian(self, xi, et, element_range=None):
        """"""
        element_range = self.___parse_element_range_CT___(element_range)
        J = dict()
        cache = dict()
        for index in element_range:
            element = self._elements[index]
            metric_signature = element.metric_signature
            if metric_signature in cache:
                J[index] = cache[metric_signature]
            else:
                m = element.ct.Jacobian(xi, et)
                J[index] = m
                cache[metric_signature] = m
        return J

    def inverse_Jacobian_matrix(self, xi, et, element_range=None):
        """"""
        element_range = self.___parse_element_range_CT___(element_range)
        iJm = dict()
        cache = dict()
        for index in element_range:
            element = self._elements[index]
            metric_signature = element.metric_signature
            if metric_signature in cache:
                iJm[index] = cache[metric_signature]
            else:
                m = element.ct.inverse_Jacobian_matrix(xi, et)
                iJm[index] = m
                cache[metric_signature] = m
        return iJm

    def inverse_Jacobian(self, xi, et, element_range=None):
        """"""
        element_range = self.___parse_element_range_CT___(element_range)
        iJ = dict()
        cache = dict()
        for index in element_range:
            element = self._elements[index]
            metric_signature = element.metric_signature
            if metric_signature in cache:
                iJ[index] = cache[metric_signature]
            else:
                m = element.ct.inverse_Jacobian(xi, et)
                iJ[index] = m
                cache[metric_signature] = m
        return iJ

    def inverse_metric_matrix(self, xi, et, element_range=None):
        """"""
        element_range = self.___parse_element_range_CT___(element_range)
        imm = dict()
        cache = dict()
        for index in element_range:
            element = self._elements[index]
            metric_signature = element.metric_signature
            if metric_signature in cache:
                imm[index] = cache[metric_signature]
            else:
                m = element.ct.inverse_metric_matrix(xi, et)
                imm[index] = m
                cache[metric_signature] = m
        return imm

    def metric_matrix(self, xi, et, element_range=None):
        """"""
        element_range = self.___parse_element_range_CT___(element_range)
        mm = dict()
        cache = dict()
        for index in element_range:
            element = self._elements[index]
            metric_signature = element.metric_signature
            if metric_signature in cache:
                mm[index] = cache[metric_signature]
            else:
                m = element.ct.metric_matrix(xi, et)
                mm[index] = m
                cache[metric_signature] = m
        return mm
