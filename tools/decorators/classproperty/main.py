# -*- coding: utf-8 -*-

from tools.decorators.classproperty.descriptor import ClassPropertyDescriptor


def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)
