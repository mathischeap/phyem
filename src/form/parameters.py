# -*- coding: utf-8 -*-
r"""
"""

from tools.frozen import Frozen
from src.config import _parse_lin_repr
from src.config import _check_sym_repr
from src.config import _global_operator_lin_repr_setting
from src.config import _global_operator_sym_repr_setting

_global_root_constant_scalars = dict()   # only cache root scalar parameters


def constant_scalar(*args):
    """make root constant scalar"""
    num_args = len(args)

    if num_args == 1:
        arg = args[0]
        if isinstance(arg, (int, float)):
            if isinstance(arg, float) and arg % 1 == 0:
                arg = int(arg)
            else:
                pass
            if arg == 1:
                sym_repr = '1'
            elif arg != 0 and (1 / arg) % 1 == 0:
                sym_repr = r'\frac{1}{' + str(int(1/arg)) + '}'
            else:
                sym_repr = str(arg)
            lin_repr = str(arg)
            is_real = True
        else:
            raise NotImplementedError()

    elif num_args == 2:
        sym_repr, lin_repr = args
        assert isinstance(sym_repr, str), f"symbolic representation must be a str."
        assert isinstance(lin_repr, str), f"symbolic representation must be a str."
        is_real = False

    else:
        raise NotImplementedError()

    # --- now we have gotten sym_repr, lin_repr and is_real ------
    assert all([_ is not None for _ in (sym_repr, lin_repr)])
    sym_repr = _check_sym_repr(sym_repr)

    for pid in _global_root_constant_scalars:
        p = _global_root_constant_scalars[pid]
        if lin_repr == p._pure_lin_repr and sym_repr == p._sym_repr:
            return p
        else:
            continue

    for pid in _global_root_constant_scalars:
        p = _global_root_constant_scalars[pid]
        assert lin_repr != p._pure_lin_repr, f"lin_repr={lin_repr} is taken."
        assert sym_repr != p._sym_repr, f"sym_repr={sym_repr} is taken."
    assert isinstance(is_real, bool), f"Almost trivial check."
    cs = ConstantScalar0Form(
        sym_repr,
        lin_repr,
        is_real,
        True,  # is_root, generate root obj only.
    )
    _global_root_constant_scalars[id(cs)] = cs
    return cs


class ConstantScalar0Form(Frozen):
    """It is actually a special scalar valued 0-form. But since it is so special,
    we do not wrapper it with a Form class
    """

    def __init__(
            self,
            sym_repr,
            lin_repr,
            is_real,
            is_root,
    ):
        """"""
        self._sym_repr = sym_repr
        if is_root:
            lin_repr, pure_lin_repr = _parse_lin_repr('scalar_parameter', lin_repr)
        else:
            pure_lin_repr = None
        self._lin_repr = lin_repr
        self._pure_lin_repr = pure_lin_repr
        self._is_real = is_real  # the pure lin repr is a numeric string.
        self._is_root = is_root
        if self._is_real:
            assert self._is_root, f"safety, almost trivial check."
            self._value = float(self._pure_lin_repr)
        else:
            self._value = None
        self._freeze()

    def pr(self):
        """print representations"""
        print(self._sym_repr, self._lin_repr)

    def __repr__(self):
        """repr"""
        real_text = 'real' if self.is_real() else 'abstract'
        super_repr = super().__repr__().split('object')[1]
        return f"<ConstantScalar0Form {real_text}: {self._lin_repr}" + super_repr

    def is_real(self):
        r"""Return True if it is a real number in \mathbb{R}."""
        return self._is_real

    def is_root(self):
        """Return True if I am a root obj."""
        return self._is_root

    def _pr_text(self):
        """"""
        if self.is_real():
            f = float(self._pure_lin_repr)
            if f == 1:
                return ''
            elif f != 0 and (1 / f) % 1 == 0:
                return r'\frac{1}{' + str(int(1/f)) + '}'
            else:
                return str(f)
        else:
            return self._sym_repr

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        """"""
        if isinstance(val, (int, float)):
            self._value = val
        else:
            assert callable(val), f"the value of a scalar parameter must be a number of a callable object."
            self._value = val

    def __call__(self, *args, **kwargs):
        """"""
        if self.is_real():
            assert isinstance(self._value, float), f'must be'
            return self.value
        elif isinstance(self.value, (int, float)):
            return self.value
        else:
            assert self.value is not None, f"pls first set a value for {self}"
            assert callable(self.value), f"if value is not a number, it must be callable."
            return self.value(*args, **kwargs)

    def __eq__(self, other):
        """self == other"""
        return self.__repr__() == other.__repr__()

    def __add__(self, other):
        """self + other"""
        if isinstance(other, (int, float)):
            if self.is_real():
                number = float(self._sym_repr) + other
                if int(number) == number:
                    number = int(number)
                else:
                    pass
                return constant_scalar(number)
            else:
                op_lin_repr = _global_operator_lin_repr_setting['plus']
                sym_repr = self._sym_repr + '+' + str(other)   # no need to check is_root.
                lin_repr = self._lin_repr + op_lin_repr + _parse_lin_repr('scalar_parameter', str(other))[0]
                return ConstantScalar0Form(sym_repr, lin_repr, False, False)

        elif other.__class__.__name__ == self.__class__.__name__:
            if other.is_real():
                number = float(other._sym_repr)
                return self + number
            else:
                op_lin_repr = _global_operator_lin_repr_setting['plus']
                sym_repr = self._sym_repr + '+' + other._sym_repr   # no need to check is_root.
                lin_repr = self._lin_repr + op_lin_repr + other._lin_repr
                return ConstantScalar0Form(sym_repr, lin_repr, False, False)

        else:
            raise NotImplementedError()

    def __radd__(self, other):
        """other + self """
        if isinstance(other, (int, float)):
            if self.is_real():
                number = float(self._sym_repr) + other
                if int(number) == number:
                    number = int(number)
                else:
                    pass
                return constant_scalar(number)
            else:
                op_lin_repr = _global_operator_lin_repr_setting['plus']
                sym_repr = str(other) + '+' + self._sym_repr   # no need to check is_root.
                lin_repr = _parse_lin_repr('scalar_parameter', str(other))[0] + op_lin_repr + self._lin_repr
                return ConstantScalar0Form(sym_repr, lin_repr, False, False)
        else:
            raise NotImplementedError()

    def __truediv__(self, other):
        """self / other"""

        if isinstance(other, (int, float)):
            return self / constant_scalar(other)
        elif other.__class__.__name__ == self.__class__.__name__:
            if self.is_real() and other.is_real():
                raise NotImplementedError()
            else:
                op_lin_repr = _global_operator_lin_repr_setting['division']
                op_sym_repr = _global_operator_sym_repr_setting['division']
                sym_repr = op_sym_repr[0] + self._sym_repr + op_sym_repr[1] + other._sym_repr + op_sym_repr[2]
                lin_repr = self._lin_repr + op_lin_repr + other._lin_repr
                return ConstantScalar0Form(sym_repr, lin_repr, False, False)

        else:
            raise Exception()


def _find_root_scalar_parameter(lin_repr):
    """"""
    for sp_id in _global_root_constant_scalars:
        sp = _global_root_constant_scalars[sp_id]
        if sp._lin_repr == lin_repr:
            return sp
        else:
            pass


def _factor_parser(factor):
    """"""
    division = _global_operator_lin_repr_setting['division']

    root_factor = _find_root_scalar_parameter(factor)

    if root_factor is not None:
        return root_factor

    elif division in factor and factor.count(division) == 1:
        factor0, factor1 = factor.split(division)
        sp0 = _find_root_scalar_parameter(factor0)
        sp1 = _find_root_scalar_parameter(factor1)
        return _Division2(sp0, sp1)

    else:
        raise NotImplementedError(factor)


class _Division2(Frozen):
    """"""

    def __init__(self, f0, f1):
        """"""
        self._f0 = f0
        self._f1 = f1
        self._freeze()

    def __call__(self, *args, **kwargs):
        """"""
        f0 = self._f0(*args, **kwargs)
        f1 = self._f1(*args, **kwargs)
        if callable(f0):
            f0 = f0()
        else:
            pass

        if callable(f1):
            f1 = f1()
        else:
            pass

        return f0 / f1

    def _pr_text(self):
        """"""
        sf0 = self._f0._sym_repr
        sf1 = self._f1._sym_repr
        return r"\frac{" + sf0 + r"}{" + sf1 + r"}"
