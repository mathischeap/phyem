# -*- coding: utf-8 -*-
r"""
"""
from msepy.tools.matrix.static.assemble import MsePyStaticLocalMatrixAssemble
from msehy.tools.matrix.static.assembled import IrregularStaticAssembledMatrix


class IrregularStaticLocalMatrixAssemble(MsePyStaticLocalMatrixAssemble):
    """"""

    @property
    def ___assembled_class___(self):
        return IrregularStaticAssembledMatrix
