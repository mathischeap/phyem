# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from copy import deepcopy


class WfDerive(Frozen):
    """"""

    def __init__(self, wf):
        self._wf = wf
        self._freeze()

    def replace(self, index, terms, signs):
        """Replace the term indicated by ``index`` by ``terms`` of ``signs``.

        """
        if isinstance(terms, (list, tuple)):
            pass
        else:
            terms = [terms, ]
            signs = [signs, ]
        assert len(terms) == len(signs), f"new terms length is not equal to the length of new signs."
        for _, sign in enumerate(signs):
            assert sign in ('+', '-'), f"{_}th sign = {sign} is wrong. Must be '+' or '-'."

        i, j, k = self._wf._parse_index(index)
        old_sign = self._wf._sign_dict[i][j][k]
        if old_sign == '+':
            signs_ = signs
        else:
            signs_ = list()
            for s_ in signs:
                signs_.append(self._switch_sign(s_))
        signs = signs_

        new_term_dict = deepcopy(self._wf._ind_dict)  # must do deepcopy, use `_ind_dict`, not a typo
        new_sign_dict = deepcopy(self._wf._ind_dict)  # must do deepcopy, use `_ind_dict`, not a typo

        new_term_dict[i][j][k] = list()
        new_sign_dict[i][j][k] = list()

        new_term_dict[i][j][k].extend(terms)
        new_sign_dict[i][j][k].extend(signs)

        new_term_dict, new_sign_dict = self._parse_new_weak_formulation_dict(new_term_dict, new_sign_dict)

        new_wf = self._wf.__class__(self._wf._test_forms, term_sign_dict=[new_term_dict, new_sign_dict])
        new_wf.unknowns = self._wf.unknowns  # pass the unknowns
        new_wf._bc = self._wf._bc   # pass the bc

        return new_wf

    def _parse_new_weak_formulation_dict(self, new_term_dict, new_sign_dict):
        """"""
        term_dict = dict()
        sign_dict = dict()

        for i in self._wf._term_dict:
            term_dict[i] = ([], [])
            sign_dict[i] = ([], [])

            for j in range(2):
                for k in range(len(self._wf._term_dict[i][j])):
                    new_term = new_term_dict[i][j][k]
                    new_sign = new_sign_dict[i][j][k]

                    if isinstance(new_term, str) and new_term in self._wf:   # this term remain untouched.
                        assert new_sign in self._wf   # trivial check.
                        _t = self._wf[new_term][1]
                        if isinstance(new_sign, str) and new_sign in self._wf:
                            sign = self._wf[new_term][0]
                        else:
                            assert new_sign in ('+', '-'), f"sign must be + or -."
                            sign = new_sign

                        term_dict[i][j].append(_t)
                        sign_dict[i][j].append(sign)

                    else:
                        assert isinstance(new_term, list) and \
                               isinstance(new_sign, list) and \
                               len(new_sign) == len(new_term), (f"Whenever we have a "
                                                                f"modification to a term, pls put it in a list.")

                        for term, sign in zip(new_term, new_sign):

                            if term._is_able_to_be_a_weak_term():
                                term_dict[i][j].append(term)
                                sign_dict[i][j].append(sign)

                            else:
                                raise NotImplementedError()

        return term_dict, sign_dict

    def rearrange(self, rearrangement):
        """Rearrange the terms."""
        if isinstance(rearrangement, dict):
            pass
        elif isinstance(rearrangement, (list, tuple)):
            assert len(rearrangement) == len(self._wf), \
                f"When provide list (or tuple) of rearrangement, we must have a list (or tuple) of length equal to " \
                f"amount of equations."
            rag_dict = dict()
            for i, rag in enumerate(rearrangement):
                assert isinstance(rag, str) or rag is None, \
                    f"rearrangement can only be represent by str or None, {i}th rearrangement = {rag} is illegal."
                rag_dict[i] = rag
            rearrangement = rag_dict
        else:
            raise Exception(f"Rearrangement indicator format wrong.")

        term_dict = dict()
        sign_dict = dict()
        for i in self._wf._term_dict:
            term_dict[i] = ([], [])
            sign_dict[i] = ([], [])

        for i in rearrangement:
            assert isinstance(i, int), f"key: {i} is not integer, pls make sure use integer as dict keys."
            assert 0 <= i < len(self._wf), f"I cannot find {i}th equation."

            ri = rearrangement[i]
            if ri is None or ri == '':
                pass

            else:
                assert isinstance(ri, str), "Use str to represent a rearrangement pls."
                if '=' not in ri:
                    ri += '='
                else:
                    pass

                # noinspection PyUnresolvedReferences
                left_terms, right_terms = ri.replace(' ', '').split('=')
                _left_terms = left_terms.replace(',', '')
                _right_terms = right_terms.replace(',', '')

                _ = _left_terms + _right_terms
                assert _.isnumeric(), \
                    f"rearrangement for {i}th equation: {ri} is illegal, using only comma to separate " \
                    f"positive indices."

                left_terms = left_terms.split(',')
                right_terms = right_terms.split(',')

                number_terms = len(self._wf._term_dict[i][0]) + len(self._wf._term_dict[i][1])

                if right_terms == [''] and len(left_terms) < number_terms:  # move all rest terms to right
                    right_terms = list()
                    for m in range(number_terms):
                        if str(m) not in left_terms:
                            right_terms.append(str(m))
                        else:
                            pass
                elif left_terms == [''] and len(right_terms) < number_terms:  # move all rest terms to left
                    left_terms = list()
                    for m in range(number_terms):
                        if str(m) not in right_terms:
                            left_terms.append(str(m))
                        else:
                            pass
                else:
                    pass

                _ = list()
                if left_terms != ['']:
                    _.extend(left_terms)
                else:
                    left_terms = 0
                if right_terms != ['']:
                    _.extend(right_terms)
                else:
                    right_terms = 0

                _ = [int(__) for __ in _]
                for _i_ in _:
                    assert _i_ in range(number_terms), \
                        f"touching wrong index: {_i_} for equation #{i} whose valid terms ranged({number_terms})."

                _.sort()
                _ = [str(__) for __ in _]

                assert _ == [str(j) for j in range(number_terms)], \
                    f'indices of rearrangement for {i}th equation: {ri} are wrong.'

                if left_terms == 0:
                    pass
                else:
                    for k in left_terms:
                        target_index = str(i) + '-' + k
                        sign, target_term = self._wf[target_index]
                        if target_term == 0:
                            continue
                        else:
                            _j = self._wf._parse_index(target_index)[1]

                            if _j == 0:    # the target term is also at left.
                                pass
                            elif _j == 1:  # the target term is at opposite side, i.e., right
                                sign = self._switch_sign(sign)
                            else:
                                raise Exception()
                            term_dict[i][0].append(target_term)
                            sign_dict[i][0].append(sign)

                if right_terms == 0:
                    pass
                else:
                    for m in right_terms:
                        target_index = str(i) + '-' + m
                        sign, target_term = self._wf[target_index]
                        if target_term == 0:
                            continue
                        else:
                            _j = self._wf._parse_index(target_index)[1]

                            if _j == 0:  # the target term is at opposite side, i.e., left
                                sign = self._switch_sign(sign)
                            elif _j == 1:  # the target term is also at right.
                                pass
                            else:
                                raise Exception()
                            term_dict[i][1].append(target_term)
                            sign_dict[i][1].append(sign)

        for _i in self._wf._term_dict:
            if _i not in rearrangement or rearrangement[_i] is None or rearrangement[_i] == '':
                term_dict[_i] = self._wf._term_dict[_i]
                sign_dict[_i] = self._wf._sign_dict[_i]
            else:
                pass

        new_wf = self._wf.__class__(self._wf._test_forms, term_sign_dict=[term_dict, sign_dict])
        new_wf.unknowns = self._wf.unknowns   # pass the unknowns
        new_wf._bc = self._wf._bc             # pass the bc
        return new_wf

    @staticmethod
    def _switch_sign(sign):
        """switch sign."""
        if sign == '+':
            return '-'
        elif sign == '-':
            return '+'
        else:
            raise Exception()

    def switch_sign(self, rows):
        """Switch the signs of all terms of equations ``rows``"""
        if isinstance(rows, int):
            rows = [rows, ]
        else:
            assert isinstance(rows, (list, tuple)), f"put rows in a list or tuple."
            rows = [int(_) for _ in rows]

        for row in rows:
            assert 0 <= row < len(self._wf), rf"row={row} is wrong."

        new_sign_dict = dict()

        for i in self._wf._term_dict:
            if i not in rows:
                new_sign_dict[i] = self._wf._sign_dict[i]
            else:
                new_sign_dict[i] = ([], [])
                old_signs = self._wf._sign_dict[i]
                left_signs, right_signs = old_signs

                for sign in left_signs:
                    new_sign_dict[i][0].append(
                        self._switch_sign(sign)
                    )
                for sign in right_signs:
                    new_sign_dict[i][1].append(
                        self._switch_sign(sign)
                    )

        new_wf = self._wf.__class__(
            self._wf._test_forms,
            term_sign_dict=[self._wf._term_dict, new_sign_dict]
        )
        new_wf.unknowns = self._wf.unknowns   # pass the unknowns
        new_wf._bc = self._wf._bc             # pass the bc
        return new_wf

    def integration_by_parts(self, index):
        """Do integration by parts for the term indicated by ``index``."""
        term = self._wf[index][1]
        terms, signs = term._integration_by_parts()
        return self.replace(index, terms, signs)

    def split(self, index, *args, **kwargs):
        """Split the term indicated by ``index`` into multiple terms."""
        term = self._wf[index][1]
        terms, signs = term.split(*args, **kwargs)
        return self.replace(index, terms, signs)

    def delete(self, index):
        """Delete the term indicated by ``index``."""
        wf = self.replace(index, [], [])
        return wf

    def commutation_wrt_inner_and_x(self, index, *args, **kwargs):
        term = self._wf[index][1]
        terms, signs = term._commutation_wrt_inner_and_x(*args, **kwargs)
        return self.replace(index, terms, signs)

    def switch_to_duality_pairing(self, index):
        term = self._wf[index][1]
        terms, signs = term._switch_to_duality_pairing()
        return self.replace(index, terms, signs)
