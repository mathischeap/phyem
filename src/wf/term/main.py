# -*- coding: utf-8 -*-
r"""
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

_global_wf_terms = dict()

from src.spaces.continuous.Lambda import ScalarValuedFormSpace
from src.spaces.continuous.bundle import BundleValuedFormSpace
from src.spaces.continuous.bundle_diagonal import DiagonalBundleValuedFormSpace

from tools.frozen import Frozen
from src.form.main import _global_forms
from src.form.others import _find_form
from src.form.operators import codifferential, d, trace, Hodge
from src.config import _global_operator_lin_repr_setting
from src.config import _wf_term_default_simple_patterns as _simple_patterns
from src.form.parameters import constant_scalar, ConstantScalar0Form
from src.config import _global_operator_sym_repr_setting
from src.config import _non_root_lin_sep
from src.wf.term.ap import _SimplePatternAPParser
from src.wf.term.pattern import _dp_simpler_pattern_examiner_scalar_valued_forms
from src.wf.term.pattern import _dp_simpler_pattern_examiner_scalar_valued_forms_restrict
from src.wf.term.pattern import _inner_simpler_pattern_examiner_scalar_valued_forms
from src.wf.term.pattern import _inner_simpler_pattern_examiner_bundle_valued_forms
from src.wf.term.pattern import _inner_simpler_pattern_examiner_diagonal_bundle_valued_forms
_cs1 = constant_scalar(1)


class _WeakFormulationTerm(Frozen):
    """Factor multiplies the term, <f0|f1> or (f0, f1).
    """

    def __init__(self, f0, f1, factor=None):
        """"""
        # cache all weak formulation terms.
        _global_wf_terms[id(self)] = self

        self._mesh = f0.mesh
        self._f0 = f0
        self._f1 = f1

        if factor is None or factor == 1:
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
        self._restrict_manifold = None
        self._freeze()

    def restrict(self, sym):
        """"""
        raise NotImplementedError()

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
        if not isinstance(info_dict, dict):
            raise Exception('put info in a dict with keys being indicators.')
        self._extra_info.update(info_dict)
        self._check_extra_info()
        self.___simple_pattern___ = None            # renew pattern for safety
        self.___simple_pattern_keys___ = None       # renew pattern for safety

    def _check_extra_info(self):
        """"""
        for indicator in self._extra_info:
            if indicator == 'known-forms':
                known_forms = self._extra_info[indicator]
                if isinstance(known_forms, (list, tuple)):
                    pass
                else:
                    known_forms = [known_forms]

                for kf in known_forms:
                    assert kf in self._efs, f"form {kf} is not a valid elementary form."

            else:
                raise Exception()

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
        from src.config import RANK, MASTER_RANK
        if RANK != MASTER_RANK:
            return
        else:
            fig = plt.figure(figsize=(5 + len(self._lin_repr)/20, 2))
            plt.axis((0, 1, 0, 1))
            plt.text(0, 0.75, 'linguistic : ' + f"{self._lin_repr}", ha='left', va='center', size=15)
            plt.text(0, 0.25, 'symbolic : ' + f"${self._sym_repr}$", ha='left', va='center', size=15)
            plt.axis('off')
            from src.config import _setting
            plt.show(block=_setting['block'])
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
        from src.form.main import Form

        if f == 'f0':
            assert by.__class__.__name__ == 'Form' and by.space == f.space, f"Spaces do not match."
            assert self._f0.space == by.space, f"spaces do not match."
            return self.__class__(by, self._f1, factor=self._factor)

        elif f == 'f1':
            assert by.__class__.__name__ == 'Form' and by.space == f.space, f"Spaces do not match."
            assert self._f1.space == by.space, f"spaces do not match."
            return self.__class__(self._f0, by, factor=self._factor)

        elif f.__class__ is Form:
            assert f.space == by.space, f"spaces do not match."
            if which == 'all':
                places = {
                    'f0': 'all',
                    'f1': 'all',
                }
            elif isinstance(which, (list, tuple)) and len(which) == 2:
                # which can be like (0, 0), (1, 2) which refer to the first one in term f0, and
                # the third one in f1.
                # Or which = (0, [0, 2, 3]), it means the first, third, forth ones in f0.
                wh0, wh1 = which
                assert wh0 in (0, 1, 'f0', 'f1'), f"which={which} format wrong."
                if wh0 == 0:
                    wh0 = 'f0'
                elif wh0 == 1:
                    wh0 = 'f1'
                else:
                    pass
                if isinstance(wh1, int):
                    assert wh1 >= 0, f"which={which} format wrong."
                    places = {
                        wh0: [wh1],
                    }
                elif isinstance(wh1, (list, tuple)):
                    for _ in wh1:
                        assert isinstance(_, int) and _ >= 0, f"which={which} format wrong."
                    places = {
                        wh0: wh1,
                    }

                else:
                    raise NotImplementedError()

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
                    raise NotImplementedError()

            if change_sign:
                sign = '-'
            else:
                sign = '+'
            new_term = self.__class__(f0, f1, factor=factor)
            if self._restrict_manifold is None:
                pass
            else:
                new_term.restrict(self._restrict_manifold._sym_repr)
            return new_term, sign
        else:
            raise NotImplementedError()

    def split(self, f, into, signs, factors=None, which=None):
        """Split `which` `f` `into` of `signs`."""
        if f in ('f0', 'f1'):
            assert which is None, f"When specify f0 or f1, no need to set which."
            term_class = self.__class__

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

            if f == 'f0':
                f1 = self._f1
                for i, ifi in enumerate(into):
                    assert signs[i] in ('+', '-'), f"{i}th sign = {signs[i]} is wrong."
                    assert ifi.__class__.__name__ == 'Form', f"{i}th object = {ifi} is not a form."
                    assert ifi.mesh == f1.mesh, f"mesh of {i}th object = {ifi.mesh} does not fit."
                    term = term_class(ifi, f1, factor=factors[i])
                    if self._restrict_manifold is None:
                        pass
                    else:
                        raise NotImplementedError()
                    new_terms.append(term)
                return new_terms, signs
            elif f == 'f1':
                f0 = self._f0
                for i, ifi in enumerate(into):
                    assert signs[i] in ('+', '-'), f"{i}th sign = {signs[i]} is wrong."
                    assert ifi.__class__.__name__ == 'Form', f"{i}th object = {ifi} is not a form."
                    assert ifi.mesh == f0.mesh, f"mesh of {i}th object = {ifi.mesh} does not fit."
                    term = term_class(f0, ifi, factor=factors[i])
                    if self._restrict_manifold is None:
                        pass
                    else:
                        raise NotImplementedError()
                    new_terms.append(term)
                return new_terms, signs
            else:
                raise Exception()

        else:
            raise NotImplementedError()

    def ap(self, **kwargs):
        """Return the algebraic proxy of this term."""

        if self._simple_pattern == '':
            raise NotImplementedError(f"Find no simple pattern for term {self}.")

        else:
            ap, sign, linearity = _SimplePatternAPParser(self)(**kwargs)
            return ap, sign, linearity


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

    def restrict(self, sub_manifold_sym_repr):
        """"""
        from src.manifold import find_manifold
        sr1 = self._f0._sym_repr
        sr2 = self._f1._sym_repr
        lr1 = self._f0._lin_repr
        lr2 = self._f1._lin_repr
        over_ = sub_manifold_sym_repr
        olr0, olr1, olr2 = _global_operator_lin_repr_setting['duality-pairing']
        sym_repr = rf'\left<\left.{sr1}\right|{sr2}\right>_' + r"{" + over_ + "}"

        sub_manifold = find_manifold(sub_manifold_sym_repr)
        lin_repr = olr0 + lr1 + olr1 + lr2 + olr2 + sub_manifold._lin_repr
        self.___sym_repr___ = sym_repr
        self.___lin_repr___ = lin_repr
        self._restrict_manifold = sub_manifold

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
            if self._restrict_manifold is None:
                return _dp_simpler_pattern_examiner_scalar_valued_forms(
                    factor, f0, f1, self.extra_info
                )
            else:
                return _dp_simpler_pattern_examiner_scalar_valued_forms_restrict(
                    factor, f0, f1, self.extra_info, self._restrict_manifold
                )

        else:
            raise NotImplementedError()

    def _commutation_wrt_inner_and_x(self, current_format, target_format, indicator_dict=None):
        """
        <A | B x C> = <B | C x A> = <C | A x B>.

        Parameters
        ----------
        current_format :
            '<A | B x C>', '<B | C x A>', '<C | A x B>' or so on
        target_format :
            '<A | B x C>', '<B | C x A>', '<C | A x B>' or so on'

        indicator_dict : None
            For example,
                indicator_dict = {
                    'A': u
                    'B': f
                    'C': w.exterior_derivative()
                }
            where keys are strings appeared in `current_format` and `target_format` and values are forms
            these strings are representing.

            If `indicator_dict` is None, then we try to parse A, B, C from the term.

        """
        assert isinstance(current_format, str), f"put current format into a string."
        assert isinstance(target_format, str), f"put current format into a string."
        current_format = current_format.replace(' ', '')   # remove space
        target_format = target_format.replace(' ', '')     # remove space
        assert len(current_format) == 7, f"current_format={current_format} wrong"
        assert len(target_format) == 7, f"target_format={target_format} wrong"
        assert current_format[0] == '<' and current_format[-1] == '>', f"current_format={current_format} wrong"
        assert target_format[0] == '<' and target_format[-1] == '>', f"target_format={target_format} wrong"
        assert '|' in current_format and current_format.count('|') == 1, f"current_format={current_format} wrong"
        assert '|' in target_format and target_format.count('|') == 1, f"target_format={target_format} wrong"
        assert '<' in current_format and current_format.count('<') == 1, f"current_format={current_format} wrong"
        assert '<' in target_format and target_format.count('<') == 1, f"target_format={target_format} wrong"
        assert '>' in current_format and current_format.count('>') == 1, f"current_format={current_format} wrong"
        assert '>' in target_format and target_format.count('>') == 1, f"target_format={target_format} wrong"
        assert 'x' in current_format and current_format.count('x') == 1, f"current_format={current_format} wrong"
        assert 'x' in target_format and target_format.count('x') == 1, f"target_format={target_format} wrong"
        assert 'X' not in current_format, f"current_format={current_format} wrong"
        assert 'X' not in target_format, f"target_format={target_format} wrong"

        current_format = current_format[1:-1]
        target_format = target_format[1:-1]

        cf0, cf1 = current_format.split('|')
        tf0, tf1 = target_format.split('|')

        if len(cf0) == 1:
            assert (len(cf1) == 3) and (cf0 not in cf1), f"current_format={current_format} wrong"
            _dict = {
                cf0: 'A',
                cf1[0]: 'B',
                cf1[2]: 'C',
            }
            normalized_cf = '<A|BxC>'
        elif len(cf1) == 1:
            assert (len(cf0) == 3) and (cf1 not in cf0), f"current_format={current_format} wrong"
            _dict = {
                cf1: 'A',
                cf0[0]: 'B',
                cf0[2]: 'C',
            }
            normalized_cf = '<BxC|A>'
        else:
            raise Exception()

        check_format = ''
        if len(tf0) == 1:
            assert len(tf1) == 3 and tf0 not in tf1, f"target_format={target_format} wrong"
            check_format += _dict[tf0] + _dict[tf1[0]] + _dict[tf1[2]]

            if check_format in ('ABC', 'BCA', 'CAB'):  # no minus
                if check_format == 'ABC':
                    normalized_tf = '<A|BxC>'
                elif check_format == 'BCA':
                    normalized_tf = '<B|CxA>'
                elif check_format == 'CAB':
                    normalized_tf = '<C|AxB>'
                else:
                    raise Exception()
                sign = '+'

            elif check_format in ('ACB', 'BAC', 'CBA'):  # no minus
                if check_format == 'ACB':
                    normalized_tf = '<A|CxB>'
                elif check_format == 'BAC':
                    normalized_tf = '<B|AxC>'
                elif check_format == 'CBA':
                    normalized_tf = '<C|BxA>'
                else:
                    raise Exception()
                sign = '-'

            else:
                raise Exception(f"cannot from {current_format} to reach {target_format}")

        elif len(tf1) == 1:
            assert len(tf0) == 3 and tf1 not in tf0, f"target_format={target_format} wrong"
            check_format += _dict[tf1] + _dict[tf0[0]] + _dict[tf0[2]]

            if check_format in ('ABC', 'BCA', 'CAB'):  # no minus
                if check_format == 'ABC':
                    normalized_tf = '<BxC|A>'
                elif check_format == 'BCA':
                    normalized_tf = '<CxA|B>'
                elif check_format == 'CAB':
                    normalized_tf = '<AxB|C>'
                else:
                    raise Exception()
                sign = '+'

            elif check_format in ('ACB', 'BAC', 'CBA'):  # no minus
                if check_format == 'ACB':
                    normalized_tf = '<CxB|A>'
                elif check_format == 'BAC':
                    normalized_tf = '<AxC|B>'
                elif check_format == 'CBA':
                    normalized_tf = '<BxA|C>'
                else:
                    raise Exception()
                sign = '-'

            else:
                raise Exception(
                    f"cannot from {current_format} to reach {target_format}")

        else:
            raise Exception()

        # ----------- make indicator dictionary ------------------------------
        normalized_indicator_dict = {}
        if indicator_dict is None:
            f0, f1 = self._f0, self._f1
            if normalized_cf == '<A|BxC>':
                normalized_indicator_dict['A'] = f0
                BC_forms = f1

            elif normalized_cf == '<BxC|A>':
                normalized_indicator_dict['A'] = f1
                BC_forms = f0

            else:
                raise Exception()

            # now we try to split BC_forms into B and C .......

            BC_lr = BC_forms._lin_repr
            BC_sr = BC_forms._sym_repr
            cross_product_lr = _global_operator_lin_repr_setting['cross_product']
            cross_product_sr = _global_operator_sym_repr_setting['cross_product']
            assert cross_product_lr in BC_lr and cross_product_sr in BC_sr, \
                f"term has no proper cross product."

            right_combinations = None

            lr_dict = {}
            for fid in _global_forms:
                f = _global_forms[fid]
                if f.is_root():
                    f_lr = f._lin_repr
                else:
                    f_lr = _non_root_lin_sep[0] + f._lin_repr + _non_root_lin_sep[1]
                lr_dict[fid] = f_lr

            for f1_id in _global_forms:
                f1 = _global_forms[f1_id]
                f1_lr = lr_dict[f1_id]
                for f2_id in _global_forms:
                    f2 = _global_forms[f2_id]
                    f2_lr = lr_dict[f2_id]

                    check_lin_repr = f1_lr + cross_product_lr + f2_lr

                    if check_lin_repr == BC_lr:
                        right_combinations = (f1._lin_repr, f2._lin_repr)
                        break
                    else:
                        pass

            if right_combinations is None:
                raise Exception(f'Do not find correct forms in {self}')
            B_lin_repr, C_lin_repr = right_combinations
            B = _find_form(B_lin_repr)
            C = _find_form(C_lin_repr)
            normalized_indicator_dict['B'] = B
            normalized_indicator_dict['C'] = C

        else:
            raise NotImplementedError()

        assert (len(normalized_indicator_dict) == 3 and
                all([_ in normalized_indicator_dict for _ in 'ABC']))

        # make a new term ------------------------------------------------------------
        A, B, C = normalized_indicator_dict['A'], normalized_indicator_dict['B'], normalized_indicator_dict['C']

        if normalized_tf == '<A|BxC>':
            new_term = DualityPairingTerm(A, B.cross_product(C), factor=self._factor)
        elif normalized_tf == '<B|CxA>':
            new_term = DualityPairingTerm(B, C.cross_product(A), factor=self._factor)
        elif normalized_tf == '<C|AxB>':
            new_term = DualityPairingTerm(C, A.cross_product(B), factor=self._factor)
        elif normalized_tf == '<A|CxB>':
            new_term = DualityPairingTerm(A, C.cross_product(B), factor=self._factor)
        elif normalized_tf == '<B|AxC>':
            new_term = DualityPairingTerm(B, A.cross_product(C), factor=self._factor)
        elif normalized_tf == '<C|BxA>':
            new_term = DualityPairingTerm(C, B.cross_product(A), factor=self._factor)

        elif normalized_tf == '<BxC|A>':
            new_term = DualityPairingTerm(B.cross_product(C), A, factor=self._factor)
        elif normalized_tf == '<CxA|B>':
            new_term = DualityPairingTerm(C.cross_product(A), B, factor=self._factor)
        elif normalized_tf == '<AxB|C>':
            new_term = DualityPairingTerm(A.cross_product(B), C, factor=self._factor)
        elif normalized_tf == '<CxB|A>':
            new_term = DualityPairingTerm(C.cross_product(B), A, factor=self._factor)
        elif normalized_tf == '<AxC|B>':
            new_term = DualityPairingTerm(A.cross_product(C), B, factor=self._factor)
        elif normalized_tf == '<BxA|C>':
            new_term = DualityPairingTerm(B.cross_product(A), C, factor=self._factor)

        else:
            raise Exception()

        return new_term, sign


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

    if s0.__class__ is ScalarValuedFormSpace and s1.__class__ is ScalarValuedFormSpace:
        assert s0.mesh == s1.mesh and s0.k == s1.k, \
            f"cannot do inner product between {f0} in {s0} and {f1} in {s1}."

    elif s0.__class__ is BundleValuedFormSpace and s1.__class__ is BundleValuedFormSpace:
        assert s0.mesh == s1.mesh and s0.k == s1.k, \
            f"cannot do inner product between {f0} in {s0} and {f1} in {s1}."

    elif s0.__class__ is DiagonalBundleValuedFormSpace and s1.__class__ is BundleValuedFormSpace:
        assert s0.mesh == s1.mesh and 0 < s1.k < s1.mesh.n

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
        s0 = f0.space
        s1 = f1.space
        super().__init__(f0, f1, factor=factor)
        if s0.__class__ is ScalarValuedFormSpace and s1.__class__ is ScalarValuedFormSpace:
            assert s0 == s1, f"spaces dis-match. {s0} and {s1}"   # mesh consistence checked here.
            over_ = self._mesh.manifold._sym_repr
        elif s0.__class__ is BundleValuedFormSpace and s1.__class__ is BundleValuedFormSpace:
            assert s0 == s1, f"spaces dis-match. {s0} and {s1}"   # mesh consistence checked here.
            over_ = self._mesh.manifold._sym_repr
        elif s0.__class__ is DiagonalBundleValuedFormSpace and s1.__class__ is BundleValuedFormSpace:
            assert s0.mesh == s1.mesh and 0 < s1.k < s1.mesh.n
            over_ = self._mesh.manifold._sym_repr
        else:
            raise NotImplementedError()

        sr1 = f0._sym_repr
        sr2 = f1._sym_repr

        lr1 = f0._lin_repr
        lr2 = f1._lin_repr

        olr0, olr1, olr2 = _global_operator_lin_repr_setting['L2-inner-product']
        sym_repr = rf'\left({sr1}, {sr2}\right)_' + r"{" + over_ + "}"
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
        if s0.__class__ is ScalarValuedFormSpace and s1.__class__ is ScalarValuedFormSpace:
            return _inner_simpler_pattern_examiner_scalar_valued_forms(factor, f0, f1, self.extra_info)
        if s0.__class__ is BundleValuedFormSpace and s1.__class__ is BundleValuedFormSpace:
            return _inner_simpler_pattern_examiner_bundle_valued_forms(factor, f0, f1, self.extra_info)
        if s0.__class__ is DiagonalBundleValuedFormSpace and s1.__class__ is BundleValuedFormSpace:
            return _inner_simpler_pattern_examiner_diagonal_bundle_valued_forms(factor, f0, f1, self.extra_info)
        else:
            raise NotImplementedError()

    def _integration_by_parts(self):
        """"""
        if _simple_patterns['(cd,)'] == self._simple_pattern:
            # we try to find the sf by testing all existing forms, this is bad. Update this in the future.
            bf = _find_form(self._f0._lin_repr, upon=codifferential)
            assert bf is not None, \
                f"something is wrong, we do not found the base form (codifferential of base form = f0)."

            # self factor must be a constant parameter.
            term_manifold = L2InnerProductTerm(bf, d(self._f1), factor=self._factor)

            if self.manifold.is_periodic():
                return [term_manifold, ], ['+', ]

            else:

                trace_form_0 = trace(Hodge(bf))

                trace_form_1 = trace(self._f1)

                term_boundary = duality_pairing(trace_form_0, trace_form_1, factor=self._factor)

                return (term_manifold, term_boundary), ('+', '-')

        else:
            raise Exception(f"Cannot apply integration by parts to "
                            f"this term of simple_pattern: {self._simple_pattern}")

    def _commutation_wrt_inner_and_x(self, current_format, target_format, indicator_dict=None):
        """
        (A, B x C) = (B, C x A) = (C, A x B).

        Parameters
        ----------
        current_format :
            '(A, B x C)', '(B, C x A)', '(C, A x B)' or so on
        target_format :
            '(A, B x C)', '(B, C x A)', '(C, A x B)' or so on'

        indicator_dict : None
            For example,
                indicator_dict = {
                    'A': u
                    'B': f
                    'C': w.exterior_derivative()
                }
            where keys are strings appeared in `current_format` and `target_format` and values are forms
            these strings are representing.

            If `indicator_dict` is None, then we try to parse A, B, C from the term.

        """
        assert isinstance(current_format, str), f"put current format into a string."
        assert isinstance(target_format, str), f"put current format into a string."
        current_format = current_format.replace(' ', '')
        target_format = target_format.replace(' ', '')
        assert len(current_format) == 7, f"current_format={current_format} wrong"
        assert len(target_format) == 7, f"target_format={target_format} wrong"
        assert current_format[0] == '(' and current_format[-1] == ')', f"current_format={current_format} wrong"
        assert target_format[0] == '(' and target_format[-1] == ')', f"target_format={target_format} wrong"
        assert ',' in current_format and current_format.count(',') == 1, f"current_format={current_format} wrong"
        assert ',' in target_format and target_format.count(',') == 1, f"target_format={target_format} wrong"
        assert '(' in current_format and current_format.count('(') == 1, f"current_format={current_format} wrong"
        assert '(' in target_format and target_format.count('(') == 1, f"target_format={target_format} wrong"
        assert ')' in current_format and current_format.count(')') == 1, f"current_format={current_format} wrong"
        assert ')' in target_format and target_format.count(')') == 1, f"target_format={target_format} wrong"
        assert 'x' in current_format and current_format.count('x') == 1, f"current_format={current_format} wrong"
        assert 'x' in target_format and target_format.count('x') == 1, f"target_format={target_format} wrong"
        assert 'X' not in current_format, f"current_format={current_format} wrong"
        assert 'X' not in target_format, f"target_format={target_format} wrong"

        current_format = current_format[1:-1]
        target_format = target_format[1:-1]

        cf0, cf1 = current_format.split(',')
        tf0, tf1 = target_format.split(',')

        if len(cf0) == 1:
            assert (len(cf1) == 3) and (cf0 not in cf1), f"current_format={current_format} wrong"
            _dict = {
                cf0: 'A',
                cf1[0]: 'B',
                cf1[2]: 'C',
            }
            normalized_cf = '(A,BxC)'
        elif len(cf1) == 1:
            assert (len(cf0) == 3) and (cf1 not in cf0), f"current_format={current_format} wrong"
            _dict = {
                cf1: 'A',
                cf0[0]: 'B',
                cf0[2]: 'C',
            }
            normalized_cf = '(BxC,A)'
        else:
            raise Exception()

        check_format = ''
        if len(tf0) == 1:
            assert len(tf1) == 3 and tf0 not in tf1, f"target_format={target_format} wrong"
            check_format += _dict[tf0] + _dict[tf1[0]] + _dict[tf1[2]]

            if check_format in ('ABC', 'BCA', 'CAB'):  # no minus
                if check_format == 'ABC':
                    normalized_tf = '(A,BxC)'
                elif check_format == 'BCA':
                    normalized_tf = '(B,CxA)'
                elif check_format == 'CAB':
                    normalized_tf = '(C,AxB)'
                else:
                    raise Exception()
                sign = '+'

            elif check_format in ('ACB', 'BAC', 'CBA'):  # no minus
                if check_format == 'ACB':
                    normalized_tf = '(A,CxB)'
                elif check_format == 'BAC':
                    normalized_tf = '(B,AxC)'
                elif check_format == 'CBA':
                    normalized_tf = '(C,BxA)'
                else:
                    raise Exception()
                sign = '-'

            else:
                raise Exception(f"cannot from {current_format} to reach {target_format}")

        elif len(tf1) == 1:
            assert len(tf0) == 3 and tf1 not in tf0, f"target_format={target_format} wrong"
            check_format += _dict[tf1] + _dict[tf0[0]] + _dict[tf0[2]]

            if check_format in ('ABC', 'BCA', 'CAB'):  # no minus
                if check_format == 'ABC':
                    normalized_tf = '(BxC,A)'
                elif check_format == 'BCA':
                    normalized_tf = '(CxA,B)'
                elif check_format == 'CAB':
                    normalized_tf = '(AxB,C)'
                else:
                    raise Exception()
                sign = '+'

            elif check_format in ('ACB', 'BAC', 'CBA'):  # no minus
                if check_format == 'ACB':
                    normalized_tf = '(CxB,A)'
                elif check_format == 'BAC':
                    normalized_tf = '(AxC,B)'
                elif check_format == 'CBA':
                    normalized_tf = '(BxA,C)'
                else:
                    raise Exception()
                sign = '-'

            else:
                raise Exception(
                    f"cannot from {current_format} to reach {target_format}")

        else:
            raise Exception()

        # ----------- make indicator dictionary ------------------------------
        normalized_indicator_dict = {}
        if indicator_dict is None:
            f0, f1 = self._f0, self._f1
            if normalized_cf == '(A,BxC)':
                normalized_indicator_dict['A'] = f0
                BC_forms = f1

            elif normalized_cf == '(BxC,A)':
                normalized_indicator_dict['A'] = f1
                BC_forms = f0

            else:
                raise Exception()

            # now we try to split BC_forms into B and C .......

            BC_lr = BC_forms._lin_repr
            BC_sr = BC_forms._sym_repr
            cross_product_lr = _global_operator_lin_repr_setting['cross_product']
            cross_product_sr = _global_operator_sym_repr_setting['cross_product']
            assert cross_product_lr in BC_lr and cross_product_sr in BC_sr, \
                f"term has no proper cross product."

            right_combinations = None

            lr_dict = {}
            for fid in _global_forms:
                f = _global_forms[fid]
                if f.is_root():
                    f_lr = f._lin_repr
                else:
                    f_lr = _non_root_lin_sep[0] + f._lin_repr + _non_root_lin_sep[1]
                lr_dict[fid] = f_lr

            for f1_id in _global_forms:
                f1 = _global_forms[f1_id]
                f1_lr = lr_dict[f1_id]
                for f2_id in _global_forms:
                    f2 = _global_forms[f2_id]
                    f2_lr = lr_dict[f2_id]

                    check_lin_repr = f1_lr + cross_product_lr + f2_lr

                    if check_lin_repr == BC_lr:
                        right_combinations = (f1._lin_repr, f2._lin_repr)
                        break
                    else:
                        pass

            if right_combinations is None:
                raise Exception(f'Do not find correct forms in {self}')
            B_lin_repr, C_lin_repr = right_combinations
            B = _find_form(B_lin_repr)
            C = _find_form(C_lin_repr)
            normalized_indicator_dict['B'] = B
            normalized_indicator_dict['C'] = C

        else:
            raise NotImplementedError()

        assert (len(normalized_indicator_dict) == 3 and
                all([_ in normalized_indicator_dict for _ in 'ABC']))

        # make a new term ------------------------------------------------------------
        A, B, C = normalized_indicator_dict['A'], normalized_indicator_dict['B'], normalized_indicator_dict['C']

        if normalized_tf == '(A,BxC)':
            new_term = L2InnerProductTerm(A, B.cross_product(C), factor=self._factor)
        elif normalized_tf == '(B,CxA)':
            new_term = L2InnerProductTerm(B, C.cross_product(A), factor=self._factor)
        elif normalized_tf == '(C,AxB)':
            new_term = L2InnerProductTerm(C, A.cross_product(B), factor=self._factor)
        elif normalized_tf == '(A,CxB)':
            new_term = L2InnerProductTerm(A, C.cross_product(B), factor=self._factor)
        elif normalized_tf == '(B,AxC)':
            new_term = L2InnerProductTerm(B, A.cross_product(C), factor=self._factor)
        elif normalized_tf == '(C,BxA)':
            new_term = L2InnerProductTerm(C, B.cross_product(A), factor=self._factor)

        elif normalized_tf == '(BxC,A)':
            new_term = L2InnerProductTerm(B.cross_product(C), A, factor=self._factor)
        elif normalized_tf == '(CxA,B)':
            new_term = L2InnerProductTerm(C.cross_product(A), B, factor=self._factor)
        elif normalized_tf == '(AxB,C)':
            new_term = L2InnerProductTerm(A.cross_product(B), C, factor=self._factor)
        elif normalized_tf == '(CxB,A)':
            new_term = L2InnerProductTerm(C.cross_product(B), A, factor=self._factor)
        elif normalized_tf == '(AxC,B)':
            new_term = L2InnerProductTerm(A.cross_product(C), B, factor=self._factor)
        elif normalized_tf == '(BxA,C)':
            new_term = L2InnerProductTerm(B.cross_product(A), C, factor=self._factor)

        else:
            raise Exception()

        return new_term, sign

    def _switch_to_duality_pairing(self):
        """(*A, B) = <A|B>"""
        f0, f1 = self._f0, self._f1
        Hodge_lin_repr = _global_operator_lin_repr_setting['Hodge']
        f0_lr = f0._lin_repr
        f1_lr = f1._lin_repr
        Hodge_len = len(Hodge_lin_repr)

        if f0_lr[:Hodge_len] == Hodge_lin_repr:  # must be (*A, B) where B can be everything like *C; (*A, *C)
            Hodge_on = f0_lr[Hodge_len:]
            other_position = 1
            other = f1
        elif f1_lr[:Hodge_len] == Hodge_lin_repr:  # must be (A, *B) where A must not be like *D; no (*D, *B)!
            Hodge_on = f1_lr[Hodge_len:]
            other_position = 0
            other = f0
        else:
            raise Exception('format is not (*A, B) or (A, *B)')

        Hodge_on_form = None   # Hodge is operated on this form.
        for fid in _global_forms:
            f = _global_forms[fid]
            if f.is_root():
                f_lr = f._lin_repr
            else:
                f_lr = _non_root_lin_sep[0] + f._lin_repr + _non_root_lin_sep[1]

            if f_lr == Hodge_on:

                Hodge_on_form = f  # Hodge is operated on this form.
                break

        if Hodge_on_form is None:
            raise Exception('Find no proper form for the Hodge operator.')
        else:
            pass

        if other_position == 1:
            new_term = DualityPairingTerm(
                Hodge_on_form, other, factor=self._factor
            )
        elif other_position == 0:
            new_term = DualityPairingTerm(
                other, Hodge_on_form, factor=self._factor
            )
        else:
            raise Exception

        return new_term, '+'
