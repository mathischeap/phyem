# -*- coding: utf-8 -*-
# noinspection PyUnresolvedReferences
r"""

.. _ap-mp:

============
Matrix proxy
============

With the fully discrete weak formulation ``wf``, we can bring it into its algebraic proxy by calling its method
``mp``, standing for *matrix proxy*,

>>> mp = wf.mp()

which is an instance of :class:`MatrixProxy`,

    .. autoclass:: MatrixProxy
        :members:

Similarly, its ``pr`` method can illustrate it properly,

>>> mp.pr()
<Figure size ...


.. _ap-ap:

========================
Algebraic representation
========================

Depend on ``mp`` is linear or nonlinear, an algebraic system can be produced
through either method ``ls`` or ``nls`` of ``mp``,
see :meth:`MatrixProxy.ls` and :meth:`MatrixProxy.nls`.

Method ``ls`` gives an instance of :class:`MatrixProxyLinearSystem`, i.e.,

    .. autoclass:: MatrixProxyLinearSystem
        :members:

And method ``nls`` leads to an instance of :class:`MatrixProxyNoneLinearSystem`, namely,

    .. autoclass:: MatrixProxyNoneLinearSystem
        :members:

In this case, ``mp`` is a linear system. Thus, we should call ``ls`` method of it,

>>> ls = mp.ls()
>>> ls.pr()
<Figure size ...

Eventually, a fully discrete abstract linear system is obtained. We can send it a particular implementation
which will *objectivize* it, for example by making matrices 2-dimensional arrays and making the vectors
1-dimensional arrays. These implementations will be introduced in the following section.

"""
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
})
matplotlib.use('TkAgg')

from src.form.others import _find_form
from src.config import _root_form_ap_vec_setting
from tools.frozen import Frozen
from src.algebra.array import AbstractArray
from src.wf.term.ap import TermLinearAlgebraicProxy
from src.wf.term.ap import TermNonLinearOperatorAlgebraicProxy
from src.algebra.linear_system import BlockMatrix, BlockColVector, LinearSystem
from src.algebra.nonlinear_system import NonLinearSystem
from src.wf.mp.linear_system import MatrixProxyLinearSystem
from src.wf.mp.nonlinear_system import MatrixProxyNoneLinearSystem


class MatrixProxy(Frozen):
    """"""

    def __init__(self, wf):
        ap = wf.ap()  # make an algebraic proxy in real time.
        if ap._fully_resolved:
            assert ap.linearity in ('linear', 'nonlinear'), \
                f"ap linearity must be linear or nonlinear when it is fully resolved."
        else:
            raise Exception(
                f"there is(are) term(s) in the wf not-resolved as 'algebraic proxy', check 'ap' of "
                f"the 'wf' to see which term(s) is(are) not resolved!."
            )
        self._wf = wf
        self._ap = ap
        self._num_equations = len(ap._term_dict)
        self._unknowns = ap.unknowns
        self._test_vectors = ap.test_vectors
        self.___total_indexing_length___ = None

        self._lbv = BlockColVector(self._num_equations)  # left block vector part
        self._rbv = BlockColVector(self._num_equations)  # right block vector part

        self._left_nonlinear_terms = [[] for _ in range(self._num_equations)]
        self._left_nonlinear_signs = [[] for _ in range(self._num_equations)]

        self._right_nonlinear_terms = [[] for _ in range(self._num_equations)]
        self._right_nonlinear_signs = [[] for _ in range(self._num_equations)]

        self._linearity = 'linear'
        for i in ap._term_dict:
            terms = ap._term_dict[i]
            signs = ap._sign_dict[i]
            for j in range(2):
                lr_terms = terms[j]
                lr_signs = signs[j]
                k = 0

                for term, sign in zip(lr_terms, lr_signs):

                    linearity, term = self._test_vector_remover(i, term)

                    if linearity == 'linear':
                        assert self._ap._linearity_dict[i][j][k] == 'linear', f'Must be this case.'
                        if j == 0:
                            self._lbv._add(i, term, sign)
                        else:
                            self._rbv._add(i, term, sign)

                    elif linearity == 'nonlinear':
                        assert self._ap._linearity_dict[i][j][k] == 'nonlinear', f'Must be this case.'
                        if self._linearity == 'linear':
                            self._linearity = 'nonlinear'
                        else:
                            pass

                        if j == 0:
                            # noinspection PyTypeChecker
                            self._left_nonlinear_terms[i].append(term)
                            self._left_nonlinear_signs[i].append(sign)
                        else:
                            # noinspection PyTypeChecker
                            self._right_nonlinear_terms[i].append(term)
                            self._right_nonlinear_signs[i].append(sign)

                    else:
                        raise NotImplementedError()

                    k += 1

        self._l_mvs = list()  # left matrix@vector sections
        self._r_mvs = list()  # right matrix@vector sections
        self.parse(self._unknowns)
        self._bc = wf._bc
        self._freeze()

    @property
    def linearity(self):
        """"""
        assert self._linearity in ('linear', 'nonlinear'), f"Linearity must be among ('linear', 'nonlinear')."
        return self._linearity

    def _test_vector_remover(self, i, term):
        """"""
        test_vector = self._test_vectors[i]

        if term.__class__ is TermLinearAlgebraicProxy:

            aa = term._abstract_array
            factor = aa._factor
            components = aa._components
            transposes = aa._transposes

            assert components[0] == test_vector, f"cannot remove test vector {test_vector} from term {term}."
            assert transposes[0] is True, f"cannot remove test vector {test_vector} from term {term}."

            new_aa = AbstractArray(
                factor=factor,
                components=components[1:],
                transposes=transposes[1:],
            )

            return 'linear', new_aa

        elif term.__class__ is TermNonLinearOperatorAlgebraicProxy:

            tf_pure_lin_repr = test_vector._pure_lin_repr[:-len(_root_form_ap_vec_setting['lin'])]
            tf = _find_form(tf_pure_lin_repr)
            if term._tf is None:
                term.set_test_form(tf)
            else:
                assert term._tf is tf, f"double check the test form."
            return 'nonlinear', term

        else:
            raise NotImplementedError(term.__class__)

    def parse(self, targets):
        """"""
        targets = list(targets)

        lbv = self._lbv
        rbv = self._rbv
        bb = BlockColVector(self._num_equations)

        for i, target in enumerate(targets):
            if target.__class__.__name__ == 'Form':
                target = target.ap()
                targets[i] = target
            bb._add(i, target, '+')

        for lor, bv in enumerate((lbv, rbv)):

            bm = BlockMatrix((self._num_equations, len(targets)))
            remaining_bv = BlockColVector(self._num_equations)

            for i, entry in enumerate(bv._entries):
                for j, term in enumerate(entry):
                    sign = bv._signs[i][j]

                    if term.__class__ is AbstractArray:

                        t = term._components[-1]
                        trans = term._transposes[-1]

                        if t in targets and not trans:  # found a correct term to be put int matrix block
                            k = targets.index(t)

                            components = term._components[:-1]
                            transposes = term._transposes[:-1]
                            factor = term._factor

                            bm_aa = AbstractArray(
                                factor=factor,
                                components=components,
                                transposes=transposes,
                            )

                            bm._add(i, k, bm_aa, sign)

                        else:

                            remaining_bv._add(i, term, sign)

                    else:
                        raise NotImplementedError()

            if lor == 0:
                if bm._is_empty():
                    pass
                else:
                    self._l_mvs.append((bm, bb))
                self._lbv = remaining_bv
            else:
                if bm._is_empty():
                    pass
                else:
                    self._r_mvs.append((bm, bb))
                self._rbv = remaining_bv

        self.___total_indexing_length___ = None

    def _total_indexing_length(self):
        """"""
        if self.___total_indexing_length___ is None:
            a = len(self._l_mvs)
            c = len(self._r_mvs)
            if self._lbv._is_empty():
                b = 0
            else:
                b = 1
            if self._rbv._is_empty():
                d = 0
            else:
                d = 1
            self.___total_indexing_length___ = (a, b, c, d), a + b + c + d

        return self.___total_indexing_length___

    def __getitem__(self, index):
        """
        To retrieve a linear term: do 'a-b' or 'a-b,c', where a, b, c are str(integer)-s.

        when index = 'a-b'
        `a` refer to the `ath` block.
        `b` refer to the `b`th entry of the ColVec of the block.

        when index = 'a-b,c'
        `a` refer to the `ath` block.
        `b,c` refer to the `b,c`th entry of the BlockMatrix of the block.

        So when `a`th block is a Matrix @ Vector, we can use either 'a-b' (vector block) or 'a-b,c' (matrix block).

        But when `a`th block is a ColVec, we can only use 'a-b'.

        To retrieve a nonlinear term, to be continued.

        """
        assert isinstance(index, str), f"pls put index in string."
        assert index.count('-') == 1, f"linear term index={index} is illegal."
        block_num, local_index = index.split('-')
        assert block_num.isnumeric(), f"linear term index={index} is illegal."
        block_num = int(block_num)
        abcd, total_length = self._total_indexing_length()
        a, b, c, d = abcd
        assert 0 <= block_num < total_length, f"linear term index={index} is illegal; it beyonds the length."
        if block_num < a:  # retrieving a term in left l_mvs
            block = self._l_mvs[block_num]
        elif block_num < a+b:  # retrieving a term in left remaining vector
            block = self._lbv
        elif block_num < a+b+c:  # retrieving a term in right l_mvs
            block_num -= a+b
            block = self._r_mvs[block_num]
        else:  # retrieving a term in right remaining vector
            block = self._rbv

        indices = eval('[' + local_index + ']')
        if isinstance(block, tuple):
            if len(indices) == 2:
                block = block[0]
            elif len(indices) == 1:
                block = block[1]
            else:
                raise Exception(f"linear term index={index} is illegal.")
        else:
            pass

        try:
            return block(*indices)
        except (IndexError, TypeError):
            raise Exception(f"linear term index={index} is illegal.")

    def _pr_text(self):
        symbolic = ''

        plus = ''
        variant = 0
        for bm_bb in self._l_mvs:
            bm, bb = bm_bb
            assert not bm._is_empty()
            if variant == 0:
                _v_plus = ''
            else:
                _v_plus = '+'
            symbolic += _v_plus
            symbolic += bm._pr_text()
            symbolic += bb._pr_text()
            variant += 1
            plus = '+'

        if self._lbv._is_empty():
            pass
        else:
            symbolic += plus + self._lbv._pr_text()

        nonlinear_text = ''
        if self.linearity == 'nonlinear':
            nonlinear_terms = self._left_nonlinear_terms
            nonlinear_signs = self._left_nonlinear_signs
            num_terms = 0
            for terms in nonlinear_terms:
                num_terms += len(terms)

            if num_terms == 0:
                pass
            else:  # there are nonlinear terms on the left hand side

                for terms, signs in zip(nonlinear_terms, nonlinear_signs):

                    if len(terms) == 0:
                        nonlinear_text += r'\boldsymbol{0}'
                    else:

                        for i, term in enumerate(terms):
                            sign = signs[i]

                            if i == 0:
                                if sign == '-':
                                    pass
                                else:
                                    sign = ''
                            else:
                                pass

                            nonlinear_text += sign + term._sym_repr

                    nonlinear_text += r'\\'

                nonlinear_text = nonlinear_text[:-len(r'\\')]

                nonlinear_text = r" + \begin{bmatrix}" + nonlinear_text + r"\end{bmatrix}"

        symbolic += nonlinear_text + '='

        plus = ''
        variant = 0
        for bm_bb in self._r_mvs:
            bm, bb = bm_bb
            assert not bm._is_empty()
            if variant == 0:
                _v_plus = ''
            else:
                _v_plus = '+'
            symbolic += _v_plus
            symbolic += bm._pr_text()
            symbolic += bb._pr_text()
            variant += 1
            plus = '+'

        if self._rbv._is_empty():
            pass
        else:
            symbolic += plus + self._rbv._pr_text()

        nonlinear_text = ''
        if self.linearity == 'nonlinear':
            nonlinear_terms = self._right_nonlinear_terms
            nonlinear_signs = self._right_nonlinear_signs
            num_terms = 0
            for terms in nonlinear_terms:
                num_terms += len(terms)

            if num_terms == 0:
                pass
            else:  # there are nonlinear terms on the left hand side

                for terms, signs in zip(nonlinear_terms, nonlinear_signs):

                    if len(terms) == 0:
                        nonlinear_text += r'\boldsymbol{0}'
                    else:

                        for i, term in enumerate(terms):
                            sign = signs[i]

                            if i == 0:
                                if sign == '-':
                                    pass
                                else:
                                    sign = ''
                            else:
                                pass

                            nonlinear_text += sign + term._sym_repr

                    nonlinear_text += r'\\'

                nonlinear_text = nonlinear_text[:-len(r'\\')]

                nonlinear_text = r" + \begin{bmatrix}" + nonlinear_text + r"\end{bmatrix}"

        symbolic += nonlinear_text

        return symbolic

    def _mp_seek_text(self):
        """seek text"""
        seek_text = self._wf._mesh.manifold._manifold_text()
        seek_text += r'seek $\left('
        form_sr_list = list()
        space_sr_list = list()
        for un in self._ap.unknowns:
            form_sr_list.append(rf' {un._sym_repr}')
            space_sr_list.append(rf"{un._shape_text()}")
        seek_text += ','.join(form_sr_list)
        seek_text += r'\right) \in '
        seek_text += r'\times '.join(space_sr_list)
        seek_text += '$, such that\n'
        return seek_text

    def pr(self, figsize=(12, 8)):
        """Print the representation, a figure, of this weak formulation.

        Parameters
        ----------
        figsize : tuple, optional
            The figure size. It has no effect when the figure is over-sized. A tight configuration will be
            applied when it is the case. The default value is ``(12, 8)``.

        """
        from src.config import RANK, MASTER_RANK
        if RANK != MASTER_RANK:
            return
        else:
            pass
        seek_text = self._mp_seek_text()
        symbolic = r"$" + self._pr_text() + r"$"
        if self._bc is None or len(self._bc) == 0:
            bc_text = ''
        else:
            bc_text = self._bc._bc_text()
        fig = plt.figure(figsize=figsize)
        plt.axis((0, 1, 0, 1))
        plt.axis('off')
        plt.text(0.05, 0.5, seek_text + symbolic + bc_text, ha='left', va='center', size=15)

        from src.config import _setting, _pr_cache
        if _setting['pr_cache']:
            _pr_cache(fig, filename='matrixProxy')
        else:
            plt.tight_layout()
            plt.show(block=_setting['block'])
        return fig

    def _parse_ls(self):
        """"""
        assert self._lbv._is_empty(), f"Format is illegal, must be like Ax=b, do '.pr()' to check!"
        assert len(self._l_mvs) == 1, f"Format is illegal, must be like Ax=b, do '.pr()' to check!"
        A, x = self._l_mvs[0]
        b = BlockColVector(self._rbv._shape)
        for Mv in self._r_mvs:
            M, v = Mv
            for i in range(M._shape[0]):
                for j in range(M._shape[1]):
                    vj, sj = v(j)
                    Mij, Sij = M(i, j)

                    assert len(vj) == len(sj) == 1, f"Format is illegal, must be like Ax=b, do pr() to check!"

                    vj, sj = vj[0], sj[0]

                    for mij, sij in zip(Mij, Sij):

                        if sj == sij:
                            sign = '+'
                        else:
                            sign = '-'

                        mij_at_vj = mij @ vj

                        b._add(i, mij_at_vj, sign)

        for i in range(self._rbv._shape):
            bi, si = self._rbv(i)
            for bj, sj in zip(bi, si):
                b._add(i, bj, sj)

        ls = LinearSystem(A, x, b)

        return ls

    def ls(self):
        """Convert self to an abstract linear system.

        Returns
        -------
        ls : :class:`MatrixProxyLinearSystem`
            The linear system instance.

        """
        assert self.linearity == 'linear', f"'mp' is not linear, try to use '.nls'."
        ls = self._parse_ls()
        return MatrixProxyLinearSystem(self, ls, self.bc)

    def nls(self):
        """Convert self to an abstract nonlinear system.

        Returns
        -------
        nls : :class:`MatrixProxyNoneLinearSystem`
            The nonlinear system instance.

        """
        assert self.linearity == 'nonlinear', f"'mp' is linear, just use '.ls'."
        ls = self._parse_ls()
        mp_ls = MatrixProxyLinearSystem(self, ls, self.bc)
        len_right_nonlinear_terms = 0
        for terms in self._right_nonlinear_terms:
            len_right_nonlinear_terms += len(terms)
        assert len_right_nonlinear_terms == 0, \
            f"To initialize a nonlinear system, move all nonlinear terms to the left hand side first."
        nls = NonLinearSystem(
            ls,
            (self._left_nonlinear_terms, self._left_nonlinear_signs),
        )
        return MatrixProxyNoneLinearSystem(self, mp_ls, nls)

    @property
    def bc(self):
        return self._bc

    def _pr_temporal_advancing(self, *args, **kwargs):
        """"""
        return self._wf._pr_temporal_advancing(*args, **kwargs)
