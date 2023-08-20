# -*- coding: utf-8 -*-
r"""

"""
from tools.decorators.classproperty.descriptor import ClassPropertyDescriptor


def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)
