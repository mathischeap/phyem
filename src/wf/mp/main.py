# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
created at: 3/24/2023 2:53 PM
"""
import sys

if './' not in sys.path:
    sys.path.append('./')
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
})
matplotlib.use('TkAgg')
from tools.frozen import Frozen
from src.algebra.array import AbstractArray
from src.wf.term.ap import TermLinearAlgebraicProxy
from src.algebra.linear_system import BlockMatrix, BlockColVector, LinearSystem
from src.wf.mp.linear_system import MatrixProxyLinearSystem


class MatrixProxy(Frozen):
    """"""

    def __init__(self, wf):
        ap = wf.ap()  # make an algebraic proxy in real time.
        self._wf = wf
        self._ap = ap
        self._num_equations = len(ap._term_dict)
        self._unknowns = ap.unknowns
        self._test_vectors = ap.test_vectors
        self.___total_indexing_length___ = None

        self._lbv = BlockColVector(self._num_equations)  # left block vector part
        self._rbv = BlockColVector(self._num_equations)  # right block vector part

        for i in ap._term_dict:
            terms = ap._term_dict[i]
            signs = ap._sign_dict[i]
            for j in range(2):
                lr_terms = terms[j]
                lr_signs = signs[j]
                for term, sign in zip(lr_terms, lr_signs):
                    term = self._test_vector_remover(i, term)
                    if j == 0:
                        self._lbv._add(i, term, sign)
                    else:
                        self._rbv._add(i, term, sign)

        self._l_mvs = list()  # left matrix@vector sections
        self._r_mvs = list()  # right matrix@vector sections
        self.parse(self._unknowns)
        self._bc = wf._bc
        self._freeze()

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
                transposes=transposes[1:]
            )

            return new_aa

        else:
            raise NotImplementedError()

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
            self.___total_indexing_length___ = (a, b, c, d), a+b+c+d
        return self.___total_indexing_length___

    def __getitem__(self, index):
        """
        when index = 'a-b'
        `a` refer to the `ath` block.
        `b` refer to the `b`th entry of the ColVec of the block.

        when index = 'a-b,c'
        `a` refer to the `ath` block.
        `b,c` refer to the `b,c`th entry of the BlockMatrix of the block.

        So when `a`th block is a Matrix @ Vector, we can use either 'a-b' (vector block) or 'a-b,c' (matrix block).

        But when `a`th block is a ColVec, we can only use 'a-b'.

        Indexing to every term.
        """
        assert isinstance(index, str), f"pls put index in string."
        assert index.count('-') == 1, f"index={index} is illegal."
        block_num, local_index = index.split('-')
        assert block_num.isnumeric(), f"index={index} is illegal."
        block_num = int(block_num)
        abcd, total_length = self._total_indexing_length()
        a, b, c, d = abcd
        assert 0 <= block_num < total_length, f"index={index} is illegal; it beyonds the length."
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
                raise Exception(f"index={index} is illegal.")
        else:
            pass

        try:
            return block(*indices)
        except (IndexError, TypeError):
            raise Exception(f"index={index} is illegal.")

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

        symbolic += '='

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
        """pr"""
        seek_text = self._mp_seek_text()
        symbolic = r"$" + self._pr_text() + r"$"
        if self._bc is None or len(self._bc) == 0:
            bc_text = ''
        else:
            bc_text = self._bc._bc_text()
        fig = plt.figure(figsize=figsize)
        plt.axis([0, 1, 0, 1])
        plt.axis('off')
        plt.text(0.05, 0.5, seek_text + symbolic + bc_text, ha='left', va='center', size=15)
        plt.tight_layout()
        from src.config import _matplot_setting
        plt.show(block=_matplot_setting['block'])
        return fig

    def ls(self):
        """convert self to an abstract linear system."""
        assert self._lbv._is_empty(), f"Format is illegal, must be like Ax=b, do pr() to check!"
        assert len(self._l_mvs) == 1, f"Format is illegal, must be like Ax=b, do pr() to check!"
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

        return MatrixProxyLinearSystem(self, ls, self.bc)

    @property
    def bc(self):
        return self._bc


if __name__ == '__main__':
    # python src/wf/mp/rct.py
    import __init__ as ph

    samples = ph.samples

    oph = samples.pde_canonical_pH(n=3, p=3)[0]
    a3, b2 = oph.unknowns
    # oph.pr()

    wf = oph.test_with(oph.unknowns, sym_repr=[r'v^3', r'u^2'])
    wf = wf.derive.integration_by_parts('1-1')
    # wf.pr(indexing=True)

    td = wf.td
    td.set_time_sequence()  # initialize a time sequence

    td.define_abstract_time_instants('k-1', 'k-1/2', 'k')
    td.differentiate('0-0', 'k-1', 'k')
    td.average('0-1', b2, ['k-1', 'k'])

    td.differentiate('1-0', 'k-1', 'k')
    td.average('1-1', a3, ['k-1', 'k'])
    td.average('1-2', a3, ['k-1/2'])
    dt = td.time_sequence.make_time_interval('k-1', 'k')

    wf = td()

    # wf.pr()

    wf.unknowns = [
        a3 @ td.time_sequence['k'],
        b2 @ td.time_sequence['k'],
    ]

    wf = wf.derive.split(
        '0-0', 'f0',
        [a3 @ td.ts['k'], a3 @ td.ts['k-1']],
        ['+', '-'],
        factors=[1/dt, 1/dt],
    )

    wf = wf.derive.split(
        '0-2', 'f0',
        [ph.d(b2 @ td.ts['k-1']), ph.d(b2 @ td.ts['k'])],
        ['+', '+'],
        factors=[1/2, 1/2],
    )

    wf = wf.derive.split(
        '1-0', 'f0',
        [b2 @ td.ts['k'], b2 @ td.ts['k-1']],
        ['+', '-'],
        factors=[1/dt, 1/dt]
    )

    wf = wf.derive.split(
        '1-2', 'f0',
        [a3 @ td.ts['k-1'], a3 @ td.ts['k']],
        ['+', '+'],
        factors=[1/2, 1/2],
    )

    wf = wf.derive.rearrange(
        {
            0: '0, 3 = 2, 1',
            1: '3, 0 = 2, 1, 4',
        }
    )

    ph.space.finite(3)

    mp = wf.mp()
    mp.parse([
        a3 @ td.time_sequence['k-1'],
        b2 @ td.time_sequence['k-1']]
    )
    mp.pr()
    ls = mp.ls()
