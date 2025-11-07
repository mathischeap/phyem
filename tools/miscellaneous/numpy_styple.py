# -*- coding: utf-8 -*-
r""""""
from types import FunctionType, MethodType


class NumpyStyleDocstringReader(object):
    """ This class can only read numpy style docstring."""
    def __init__(self, fm2read):
        isinstance(fm2read, (FunctionType, MethodType)), " <DocstringReader> : I need a function or method to read."
        self._fm2read_ = fm2read
        self._docstring_ = fm2read.__doc__
        self._docstring_no_space_ = self._docstring_.replace(' ', '')
        self.___Parameters___ = None
        self.___Returns___ = None
        self.___AllowedLinearGlobalSystem___ = None

    @property
    def Parameters(self):
        if self.___Parameters___ is None:
            _Para_ = self._docstring_no_space_.split('Parameters')[1]

            if '：' in _Para_:
                _Para_ = _Para_.replace('：', ":")

            _Para_ = _Para_.split('----------')[1]
            _Para_ = _Para_.split('\n\n')[0]
            _paras_ = _Para_.split(':')
            para_names = ()
            for ps in range(1, len(_paras_)):
                __ = _paras_[ps - 1].split('\n')[-1]
                if __[0] == '\t':
                    __ = __[1:]
                para_names += (__,)

            self.___Parameters___ = para_names

        return self.___Parameters___

    @property
    def Returns(self):
        if self.___Returns___ is None:
            _Re_ = self._docstring_no_space_.split('Returns')[1]

            if '：' in _Re_:
                _Re_ = _Re_.replace('：', ":")

            _Re_ = _Re_.split('-------')[1]
            _Re_ = _Re_.split('\n\n')[0]
            _Res_ = _Re_.split(':')
            Re_names = ()
            for rs in range(1, len(_Res_)):
                __ = _Res_[rs - 1].split('\n')[-1]
                if __[0] == '\t':
                    __ = __[1:]
                Re_names += (__,)

            self.___Returns___ = Re_names

        return self.___Returns___
