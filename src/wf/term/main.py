# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
created at: 3/2/2023 3:06 PM
"""
import sys

if './' not in sys.path:
    sys.path.append('./')

import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "text.latex.preamble": r"\usepackage{amsmath}"
})
matplotlib.use('TkAgg')

from tools.frozen import Frozen
from src.form.others import _find_form
from src.form.operators import codifferential, d, trace, Hodge
from src.config import _global_operator_lin_repr_setting
from src.config import _wf_term_default_simple_patterns as _simple_patterns
from src.form.parameters import constant_scalar, ConstantScalar0Form
from src.config import _global_operator_sym_repr_setting
from src.config import _non_root_lin_sep
from src.wf.term.ap import _SimplePatternAPParser
from src.wf.term.pattern import _dp_simpler_pattern_examiner_scalar_valued_forms
from src.wf.term.pattern import _inner_simpler_pattern_examiner_scalar_valued_forms
_cs1 = constant_scalar(1)


class _WeakFormulationTerm(Frozen):
    """Factor multiplies the term, <f0|f1> or (f0, f1).
    """

    def __init__(self, f0, f1, factor=None):
        """"""
        self._mesh = f0.mesh
        self._f0 = f0
        self._f1 = f1

        if factor is None:
            self._factor = _cs1
        elif isinstance(factor, (int, float)):
            self._factor = constant_scalar(factor)
        elif factor.__class__ is ConstantScalar0Form:
            self._factor = factor
        else:
            raise NotImplementedError(f'f{factor}')

        self._efs = set()
        self._efs.update(f0.elementary_forms)
        self._efs.update(f1.elementary_forms)
        self.___sym_repr___ = None
        self.___lin_repr___ = None
        self.___simple_pattern___ = None
        self.___simple_pattern_keys___ = None
        self._extra_info = dict()
        self._freeze()

    @property
    def _simple_pattern(self):
        if self.___simple_pattern___ is None:
            self.___simple_pattern___, self.___simple_pattern_keys___ = \
                self._simpler_pattern_examiner(self._factor, self._f0, self._f1)

            if self.___simple_pattern___ == '':  # find no simple pattern.
                assert self.___simple_pattern_keys___ is None, "no simple pattern keys."
            else:
                assert self.___simple_pattern___ in _simple_patterns.values(), \
                    f"found unknown simple pattern: {self.___simple_pattern___}."
        return self.___simple_pattern___

    def _simpler_pattern_examiner(self, factor, f0, f1):
        """"""
        raise NotImplementedError()

    @property
    def extra_info(self):
        """to store extra info for, for example, parse patterns."""
        return self._extra_info

    def add_extra_info(self, info_dict):
        """"""
        self._extra_info.update(info_dict)

    @property
    def _sym_repr(self):
        if self._factor == _cs1:
            return self.___sym_repr___
        else:
            if self._factor.is_root():
                return self._factor._sym_repr + self.___sym_repr___
            else:
                frac = _global_operator_sym_repr_setting['division'][0]
                if self._factor._sym_repr[:len(frac)] == frac:
                    return self._factor._sym_repr + self.___sym_repr___
                else:
                    return r'\left(' + self._factor._sym_repr + r'\right)' + self.___sym_repr___

    @property
    def _lin_repr(self):
        if self._factor == _cs1:
            return self.___lin_repr___
        else:
            if self._factor.is_root():
                return self._factor._lin_repr + _global_operator_lin_repr_setting['multiply'] + self.___lin_repr___
            else:
                return _non_root_lin_sep[0] + self._factor._lin_repr + _non_root_lin_sep[1] + \
                    _global_operator_lin_repr_setting['multiply'] + self.___lin_repr___

    @property
    def mesh(self):
        """The mesh."""
        return self._mesh

    @property
    def manifold(self):
        return self._mesh.manifold

    @property
    def elementary_forms(self):
        return self._efs

    @staticmethod
    def _is_able_to_be_a_weak_term():
        return True

    @staticmethod
    def _is_real_number_valued():
        return True

    def pr(self):
        """Print the representations of this term."""
        fig = plt.figure(figsize=(5 + len(self._lin_repr)/20, 2))
        plt.axis([0, 1, 0, 1])
        plt.text(0, 0.75, 'linguistic : ' + f"{self._lin_repr}", ha='left', va='center', size=15)
        plt.text(0, 0.25, 'symbolic : ' + f"${self._sym_repr}$", ha='left', va='center', size=15)
        plt.axis('off')
        from src.config import _matplot_setting
        plt.show(block=_matplot_setting['block'])
        return fig

    def replace(self, f, by, which='all', change_sign=False):
        """replace form `f` in this term by `by`,
        if there are more than one `f` found, apply the replacement to `which`.
        If there are 'f' in this term, which should be int or a list of int which indicating
        `f` according the sequence of `f._lin_repr` in `self._lin_repr`.


        Parameters
        ----------
        f
        by
        which
        change_sign

        Returns
        -------

        """
        if f == 'f0':
            assert by.__class__.__name__ == 'Form' and by.space == f.space, f"Spaces do not match."
            assert self._f0.space == by.space, f"spaces do not match."
            return self.__class__(by, self._f1, factor=self._factor)

        elif f == 'f1':
            assert by.__class__.__name__ == 'Form' and by.space == f.space, f"Spaces do not match."
            assert self._f1.space == by.space, f"spaces do not match."
            return self.__class__(self._f0, by, factor=self._factor)

        elif f.__class__.__name__ == 'Form':
            assert f.space == by.space, f"spaces do not match."
            if which == 'all':
                places = {
                    'f0': 'all',
                    'f1': 'all',
                }
            else:
                raise NotImplementedError(f"which={which} is not implemented.")

            f0 = self._f0
            f1 = self._f1
            factor = self._factor

            for place in places:
                if place == 'f0':
                    f0 = self._f0.replace(f, by, which=places['f0'])
                elif place == 'f1':
                    f1 = self._f1.replace(f, by, which=places['f1'])
                else:
                    raise NotImplementedError
            if change_sign:
                sign = '-'
            else:
                sign = '+'
            return self.__class__(f0, f1, factor=factor), sign

        else:
            raise NotImplementedError()

    def split(self, f, into, signs, factors=None, which=None):
        """Split `which` `f` `into` of `signs`."""
        if f in ('f0', 'f1'):
            assert which is None, f"When specify f0 or f1, no need to set `which`."
            term_class = self.__class__
            f1 = self._f1
            assert isinstance(into, (list, tuple)), f"put split objects into a list or tuple even there is only one."
            assert len(into) >= 1, f"number of split objects must be equal to or larger than 1."
            assert len(into) == len(signs), f"objects and signs length dis-match."
            if factors is None:
                factors = [1 for _ in range(len(into))]
            else:
                if not isinstance(factors, (list, tuple)):
                    factors = [factors, ]
                else:
                    pass
            assert len(signs) == len(factors), f"signs and factors length dis-match."

            new_terms = list()
            for i, ifi in enumerate(into):
                assert signs[i] in ('+', '-'), f"{i}th sign = {signs[i]} is wrong."
                assert ifi.__class__.__name__ == 'Form', f"{i}th object = {ifi} is not a form."
                assert ifi.mesh == f1.mesh, f"mesh of {i}th object = {ifi.mesh} does not fit."
                term = term_class(ifi, f1, factor=factors[i])
                new_terms.append(term)
            return new_terms, signs

        else:
            raise NotImplementedError()

    def ap(self, **kwargs):
        """Return the algebraic proxy of this term."""

        if self._simple_pattern == '':
            raise NotImplementedError(f"Find no simple pattern for term {self}.")

        else:
            ap, sign = _SimplePatternAPParser(self)(**kwargs)

        return ap, sign


def duality_pairing(f0, f1, factor=None):
    """

    Parameters
    ----------
    f0
    f1
    factor

    Returns
    -------

    """
    s1 = f0.space
    s2 = f1.space
    if s1.__class__.__name__ == 'ScalarValuedFormSpace' and s2.__class__.__name__ == 'ScalarValuedFormSpace':
        assert s1.mesh == s2.mesh and s1.k + s2.k == s1.mesh.ndim, \
            f"cannot do duality pairing between {f0} in {s1} and {f1} in {s2}."
    else:
        raise Exception(f'cannot do duality pairing between {f0} in {s1} and {f1} in {s2}.')
    return DualityPairingTerm(f0, f1, factor=factor)


class DualityPairingTerm(_WeakFormulationTerm):
    """

    Parameters
    ----------
    f0
    f1
    """

    def __init__(self, f0, f1, factor=None):
        """

        Parameters
        ----------
        f0
        f1
        """
        s0 = f0.space
        s1 = f1.space
        super().__init__(f0, f1, factor=factor)
        if s0.__class__.__name__ == 'ScalarValuedFormSpace' and s1.__class__.__name__ == 'ScalarValuedFormSpace':
            assert s0.mesh == s1.mesh and s0.k + s1.k == s0.mesh.ndim, \
                f"cannot do duality pairing between {f0} in {s0} and {f1} in {s1}."
            over_ = self._mesh.manifold._sym_repr
        else:
            raise Exception(f'cannot do duality pairing between {f0} in {s0} and {f1} in {s1}.')

        sr1 = f0._sym_repr
        sr2 = f1._sym_repr

        lr1 = f0._lin_repr
        lr2 = f1._lin_repr

        olr0, olr1, olr2 = _global_operator_lin_repr_setting['duality-pairing']
        sym_repr = rf'\left<\left.{sr1}\right|{sr2}\right>_' + r"{" + over_ + "}"
        lin_repr = olr0 + lr1 + olr1 + lr2 + olr2 + self.mesh.manifold._lin_repr
        self.___sym_repr___ = sym_repr
        self.___lin_repr___ = lin_repr

    def __repr__(self):
        """"""
        super_repr = super().__repr__().split('object')[1]
        return '<Duality Pairing ' + self._sym_repr + f'{super_repr}'

    def _simpler_pattern_examiner(self, factor, f0, f1):
        """"""
        s0 = f0.space
        s1 = f1.space
        if s0.__class__.__name__ == 'ScalarValuedFormSpace' and \
                s1.__class__.__name__ == 'ScalarValuedFormSpace':
            return _dp_simpler_pattern_examiner_scalar_valued_forms(factor, f0, f1)
        else:
            return tuple()


def inner(f0, f1, factor=None, method='L2'):
    """

    Parameters
    ----------
    f0
    f1
    factor
    method

    Returns
    -------

    """

    if f0.__class__.__name__ == 'Form' or f1.__class__.__name__ == 'Form':
        pass
    else:
        raise NotImplementedError()

    s0 = f0.space
    s1 = f1.space

    if s0.__class__.__name__ == 'ScalarValuedFormSpace' and s1.__class__.__name__ == 'ScalarValuedFormSpace':
        assert s0.mesh == s1.mesh and s0.k == s1.k, \
            f"cannot do inner product between {f0} in {s0} and {f1} in {s1}."
    else:
        raise Exception(f'cannot do inner product between {f0} in {s0} and {f1} in {s1}.')

    if method == 'L2':
        return L2InnerProductTerm(f0, f1, factor=factor)
    else:
        raise NotImplementedError()


class L2InnerProductTerm(_WeakFormulationTerm):
    """

    Parameters
    ----------
    f0
    f1
    """

    def __init__(self, f0, f1, factor=None):
        """

        Parameters
        ----------
        f0
        f1
        """
        s1 = f0.space
        s2 = f1.space
        super().__init__(f0, f1, factor=factor)
        if s1.__class__.__name__ == 'ScalarValuedFormSpace' and s2.__class__.__name__ == 'ScalarValuedFormSpace':
            assert s1 == s2, f"spaces dis-match. {s1} and {s2}"   # mesh consistence checked here.
            over_ = self._mesh.manifold._sym_repr
        else:
            raise NotImplementedError()

        sr1 = f0._sym_repr
        sr2 = f1._sym_repr

        lr1 = f0._lin_repr
        lr2 = f1._lin_repr

        olr0, olr1, olr2 = _global_operator_lin_repr_setting['L2-inner-product']
        sym_repr = rf'\left({sr1},{sr2}\right)_' + r"{" + over_ + "}"
        lin_repr = olr0 + lr1 + olr1 + lr2 + olr2 + self.mesh.manifold._lin_repr

        self.___sym_repr___ = sym_repr
        self.___lin_repr___ = lin_repr

    def __repr__(self):
        """"""
        super_repr = super().__repr__().split('object')[1]
        return '<L2IP ' + self._sym_repr + f'{super_repr}'

    def _simpler_pattern_examiner(self, factor, f0, f1):
        """"""
        s0 = f0.space
        s1 = f1.space
        if s0.__class__.__name__ == 'ScalarValuedFormSpace' and \
                s1.__class__.__name__ == 'ScalarValuedFormSpace':
            return _inner_simpler_pattern_examiner_scalar_valued_forms(factor, f0, f1, self._extra_info)
        else:
            return tuple()

    def _integration_by_parts(self):
        """"""
        if _simple_patterns['(cd,)'] == self._simple_pattern:
            # we try to find the sf by testing all existing forms, this is bad. Update this in the future.
            bf = _find_form(self._f0._lin_repr, upon=codifferential)
            assert bf is not None, f"something is wrong, we do not found the base form " \
                                   f"(codifferential of base form = f0)."

            # self factor must be a constant parameter.
            term_manifold = L2InnerProductTerm(bf, d(self._f1), factor=self._factor)

            if self.manifold.is_periodic():
                return [term_manifold, ], ['+', ]

            else:

                trace_form_0 = trace(Hodge(bf))

                trace_form_1 = trace(self._f1)

                term_boundary = duality_pairing(trace_form_0, trace_form_1)

                return (term_manifold, term_boundary), ('+', '-')

        else:
            raise Exception(f"Cannot apply integration by parts to "
                            f"this term of simple_pattern: {self._simple_pattern}")
