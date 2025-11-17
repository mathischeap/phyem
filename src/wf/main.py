# -*- coding: utf-8 -*-
# noinspection PyUnresolvedReferences
r"""

.. _docs-wf:

================
Weak formulation
================

Once the PDE is complete, as well as its boundary conditions are correctly imposed, it shall be
tested with test function spaces through :meth:`src.pde.PartialDifferentialEquations.test_with`,

>>> wf = pde.test_with([Out2, Out1], sym_repr=['p', 'q'])

which will give an instance of :class:`WeakFormulation`.

    .. autoclass:: WeakFormulation
        :members: unknowns, pr, bc, td, mp, derive

To have a glance at this raw weak formulation, just do

>>> wf.pr()
<Figure size ...

The following figure should pop up.

.. _docs-wf-fig-wf:

.. figure:: images/docs_raw_wf.png
    :align: center
    :width: 100%

    *pr* of the weak formulation

In Fig. :any:`docs-wf-fig-wf`, we can see that boundary conditions, unknowns of the PDE are correctly inherited,
and indices of terms, ``'0-0'``, ``'0-1'`` and so on, are shown by default.


.. _docs-wf-derivations:

===========
Derivations
===========

Derivations usually should be applied to the raw weak formulation such that it could be discretized properly
later on. All possible derivations are wrapped into a class, :class:`WfDerive`,

    .. autoclass:: WfDerive
        :members:

For example, we performe integration by parts to term of index ``'1-1'``, i.e., the second term of the
second equation; :math:`\left(\mathrm{d}^\ast\tilde{\alpha}, q\right)_{\mathcal{M}}`, see Fig. :any:`docs-wf-fig-wf`,

>>> wf = wf.derive.integration_by_parts('1-1')  # integrate the term '1-1' by parts
>>> wf.pr()
<Figure size ...

We can see this replaces the original term by two new terms indexed ``'1-1'`` and ``'1-2'``. Then we do

>>> wf = wf.derive.rearrange(
...     {
...         0: '0, 1 = ',    # do nothing to the first equations; can be removed
...         1: '0, 1 = 2',   # rearrange the second equations
...     }
... )

This ``rearrange`` method does not touch the first equation, and moves the third term of the second
equation (i.e. the term indexed ``'1-2'``; the boundary integral term) to the right hand side of the equation.
To check it,

>>> wf.pr()
<Figure size ...

Please play with ``rearrange`` (together with ``pr``) untill you fully understand how it works.
How to use :meth:`WfDerive.delete` and :meth:`WfDerive.switch_sign` is obvious,

>>> _wf1 = wf.derive.delete('0-0')      # delete the first term of the first equation
>>> _wf1.pr()
<Figure size ...
>>> _wf1 = _wf1.derive.switch_sign(1)   # switch signs in the second equation
>>> _wf1.pr()
<Figure size ...

Note that these four lines of commands did not make changs to ``wf`` with which we will keep working.
And usage of :meth:`WfDerive.replace` and :meth:`WfDerive.split` will be demonstrated in
:ref:`docs-temporal-discretization`.


.. _docs-discretization:

==============
Discretization
==============

.. _docs-temporal-discretization:

Temporal discretization
=======================

Before the temporal discretization, we shall first set up an abstract time sequence,

>>> ts = ph.time_sequence()

Then we can define a time interval by

>>> dt = ts.make_time_interval('k-1', 'k', sym_repr=r'\Delta t')

This gives a time interval ``dt`` which symbolically is

.. math::

    \Delta t = t^{k} - t^{k-1}.

The temporal discretization is wrapped into property :attr:`WeakFormulation.td`
which is an instance of the wrapper class,
:class:`TemporalDiscretization`,

    .. autoclass:: TemporalDiscretization
        :members:

Pick up the temporal discrezation, ``td``, of the weak formulation ``wf`` by

>>> td = wf.td

We can set the time sequence of the temporal discretization to be the time sequence we have
defined, ``ts``,

>>> td.set_time_sequence(ts)

The varilbes will be discretized to particular abstract time instants.
For example, we do

>>> td.define_abstract_time_instants('k-1', 'k-1/2', 'k')

This command defines three abstract time instants, i.e.,

.. math::

    t^{k-1},\quad t^{k-\frac{1}{2}}, \quad t^{k}, \quad k\in\left\lbrace 1,2,3,\cdots\right\rbrace.

Now, we can do the temporal discretization. For example, we apply an implicit midpoint discretization to
the weak formulation, we shall do

>>> td.differentiate('0-0', 'k-1', 'k')
>>> td.average('0-1', b, ['k-1', 'k'])
>>> td.differentiate('1-0', 'k-1', 'k')
>>> td.average('1-1', a, ['k-1', 'k'])
>>> td.average('1-2', a, ['k-1/2'])

where, at time step from :math:`t^{k-1}` to :math:`t^{k}`,
i) ``differentiate`` method is applied to terms indexed ``'0-0'`` and ``'1-0'``,
i.e. the time derivative terms,

.. math::

    \left(\partial_t \tilde{\alpha}, p\right)_\mathcal{M}
    \quad\text{and}\quad
    \left(\partial_t \tilde{\beta}, q\right)_\mathcal{M},

and ii) ``average`` method is applied to terms indexed ``'0-1'``, ``'1-1'`` and ``'1-2'``,
i.e.,

.. math::

    - \left(\mathrm{d} \tilde{\beta}, p\right)_\mathcal{M}
    \quad\text{and}\quad
    \left(\tilde{\alpha}, \mathrm{d} q\right)_\mathcal{M}
    \quad\text{and}\quad
    \left< \left.\mathrm{tr} \left(\star\tilde{\alpha}\right)\right| \mathrm{tr} q\right>_{\partial\mathcal{M}}.

To let all these temporal discretization take effects, we just need to call the ``td`` property, i.e.,

>>> wf = td()
>>> wf.pr()
<Figure size ...

The returned object is a new weak formulation instance which has received the desired temporal discretization.
We shall set the unknowns of this new weak formulation by

>>> wf.unknowns = [
...     a @ ts['k'],
...     b @ ts['k']
... ]

which means the unknowns will be

.. math::

    \left.\tilde{\alpha}\right|^{k} \quad \text{and}\quad \left.\tilde{\beta}\right|^{k},

i.e. :math:`\tilde{\alpha}(\Omega, t^k)` and :math:`\tilde{\beta}(\Omega, t^k)`.

We now need to split the composite terms into separate ones. This can be done through ``split`` method
of ``derive`` property,

>>> wf = wf.derive.split(
...     '0-0', 'f0',
...     [a @ ts['k'], a @ ts['k-1']],
...     ['+', '-'],
...     factors=[1/dt, 1/dt],
... )
>>> wf.pr()
<Figure size ...

This will split the first entry
(indicated by ``'f0'`` considering the inner product term is
:math:`\left(\text{f0},\text{f1}\right)_{\mathcal{M}}`)
of the tern indexed by ``'0-0'`` into two new terms, as explained by the remaining inputs,

.. math::

    + \dfrac{1}{\Delta t}\left.\tilde{\alpha}\right|^k
    \quad \text{and} \quad
    - \dfrac{1}{\Delta t} \left.\tilde{\alpha}\right|^{k-1}.

The ``pr`` output should have also explained everything clearly.

.. note::

    Note that after each particular method call of
    ``derive``, a new weak formulation is returned;
    the indexing system is renewed. Thus, carefully check out the
    indexing system befoew any further derivations.

Keep splitting the remian composite terms,

>>> wf = wf.derive.split(
...     '1-0', 'f0',
...     [b @ ts['k'], b @ ts['k-1']],
...     ['+', '-'],
...     factors=[1/dt, 1/dt],
... )

>>> wf = wf.derive.split(
...     '0-2', 'f0',
...     [(b @ ts['k']).exterior_derivative(), (b @ ts['k-1']).exterior_derivative()],
...     ['+', '+'],
...     factors=[1/2, 1/2],
... )

>>> wf = wf.derive.split(
...     '1-2', 'f0',
...     [(a @ ts['k']), (a @ ts['k-1'])],
...     ['+', '+'],
...     factors=[1/2, 1/2],
... )

Then we should rearrange the terms,

>>> wf = wf.derive.rearrange(
...     {
...         0: '0, 2 = 1, 3',
...         1: '2, 0 = 3, 1, 4',
...      }
... )
>>> wf.pr()
<Figure size ...

We now obtain the final (semi-)discrete system for the linear port-Hamiltonian system.

.. _docs-spacial-discretization:

Spacial discretization
=======================

Since we are already working with an abstract mesh, the spacial discretization can be accomplished
simply by specifying finite degrees to finite dimensional forms we have made.
This can be done globally by using

>>> ph.space.finite(3)

which specifies degrees of all finite dimensonal forms to 3. You can also set the degree of
an individual form through its ``degree`` property, see :attr:`src.form.main.Form.degree`. Now if you
check the ``pr`` output, you will see the degrees of the forms are correctly reflected by the spaces
they are in. For example,

>>> wf.pr()
<Figure size ...

We are ready to bring this weak formulation into its algebraic proxy (linear algebraic form) now.

"""
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "text.latex.preamble": r"\usepackage{amsmath, amssymb}",
})
matplotlib.use('TkAgg')

from phyem.tools.frozen import Frozen
from phyem.src.wf.td import TemporalDiscretization
from phyem.src.bc import BoundaryCondition
from phyem.src.wf.derive import WfDerive
from phyem.src.wf.ap.main import AlgebraicProxy
from phyem.src.wf.mp.main import MatrixProxy
from phyem.src.config import _pde_test_form_lin_repr
from phyem.src.config import _form_evaluate_at_repr_setting


class WeakFormulation(Frozen):
    """The Weak Formulation class."""

    def __init__(
            self, test_forms,
            term_sign_dict=None,
            expression=None,
            interpreter=None,
            wfs=None,
            merge=None,
    ):
        """

        Parameters
        ----------
        test_forms
        term_sign_dict
        expression
        interpreter
        wfs
        merge

        """
        if term_sign_dict is not None:  # initialize a weak formulation from term sign dict.
            assert expression is None
            assert wfs is None
            assert merge is None
            self._parse_term_sign_dict(term_sign_dict, test_forms)

        elif expression is not None:
            assert term_sign_dict is None
            assert wfs is None
            assert merge is None
            self._parse_expression(expression, interpreter, test_forms)

        elif merge is not None:  # merge multiple weak formulations
            assert term_sign_dict is None
            assert expression is None
            assert wfs is None
            self._initialize_through_merging(merge, test_forms)

        elif wfs is not None:
            # provided multiple weak forms, we merge them.
            assert test_forms is None
            assert term_sign_dict is None
            assert expression is None
            assert merge is None
            self._merge_multiple_weak_formulations(wfs)

        else:
            raise Exception()

        self._meshes, self._mesh = self._parse_meshes(self._term_dict)
        self._consistence_checker()
        self._unknowns = None
        self._derive = None
        self._bc = None
        self._terms = None
        self._freeze()

    def _parse_term_sign_dict(self, term_sign_dict, test_forms):
        """"""
        term_dict, sign_dict = term_sign_dict
        ind_dict = dict()
        indexing = dict()
        num_eq = len(term_dict)
        for i in range(num_eq):   # ith equation
            assert i in term_dict and i in sign_dict, f"numbering of equations must be 0, 1, 2, ..."
            ind_dict[i] = ([], [])
            k = 0
            for j, terms in enumerate(term_dict[i]):
                for m in range(len(terms)):
                    index = str(i) + '-' + str(k)
                    k += 1
                    indexing[index] = (sign_dict[i][j][m], term_dict[i][j][m])
                    ind_dict[i][j].append(index)

        self._test_forms = test_forms
        self._term_dict = term_dict
        self._sign_dict = sign_dict
        self._ind_dict = ind_dict
        self._indexing = indexing

    def _parse_expression(self, expression, interpreter, test_forms):
        """"""
        term_dict = dict()
        sign_dict = dict()
        ind_dict = dict()
        indexing = dict()
        for i, equation in enumerate(expression):

            equation = equation.replace(' ', '')  # remove all spaces
            equation = equation.replace('-', '+-')  # let all terms be connected by +

            term_dict[i] = ([], [])  # for left terms and right terms of ith equation
            sign_dict[i] = ([], [])  # for left terms and right terms of ith equation
            ind_dict[i] = ([], [])  # for left terms and right terms of ith equation

            k = 0
            for j, lor in enumerate(equation.split('=')):
                local_terms = lor.split('+')

                for loc_term in local_terms:
                    if loc_term == '' or loc_term == '-':  # found empty terms, just ignore.
                        pass
                    else:
                        if loc_term == '0':
                            pass
                        else:
                            if loc_term[0] == '-':
                                assert loc_term[1:] in interpreter, f"found term {loc_term[1:]} not interpreted."
                                sign = '-'
                                term = interpreter[loc_term[1:]]
                            else:
                                assert loc_term in interpreter, f"found term {loc_term} not interpreted"
                                sign = '+'
                                term = interpreter[loc_term]

                            sign_dict[i][j].append(sign)
                            term_dict[i][j].append(term)
                            index = str(i) + '-' + str(k)
                            k += 1
                            indexing[index] = (sign, term)
                            ind_dict[i][j].append(index)

        self._test_forms = test_forms
        self._term_dict = term_dict
        self._sign_dict = sign_dict
        self._ind_dict = ind_dict
        self._indexing = indexing

    def _initialize_through_merging(self, merge, test_forms):
        """"""
        term_dict = dict()
        sign_dict = dict()
        no = 0
        for i in merge:
            eqi = merge[i]
            if eqi.__class__.__name__ == 'PartialDifferentialEquations':
                tdi, sdi = eqi._term_dict, eqi._sign_dict
                for j in tdi:
                    term_dict[no] = tdi[j]
                    sign_dict[no] = sdi[j]
                    no += 1

            elif isinstance(eqi, dict):
                tdi, sdi = eqi['_term_dict'], eqi['_sign_dict']
                term_dict[no] = tdi
                sign_dict[no] = sdi
                no += 1

            else:
                raise NotImplementedError()

        for i in term_dict:
            for j, terms in enumerate(term_dict[i]):
                for k, term in enumerate(terms):
                    assert term._is_able_to_be_a_weak_term, f"term[{i}][{j}][{k}] = {term} " \
                                                            f"is not suitable for a weak formulation."
                    sign = sign_dict[i][j][k]
                    assert sign in ('+', '-'), f"sign[{i}][{j}][{k}] = {sign} illegal."
        self._parse_term_sign_dict([term_dict, sign_dict], test_forms)

    def _merge_multiple_weak_formulations(self, wfs):
        """"""
        test_forms = []
        for wf in wfs:
            assert wf.__class__ is self.__class__
            test_forms.extend(wf.test_forms)

        term_dict, sign_dict = {}, {}

        i = 0
        for wf in wfs:
            for j in wf._term_dict:

                td = wf._term_dict[j]
                sd = wf._sign_dict[j]

                term_dict[i] = td
                sign_dict[i] = sd

                i += 1

        self._parse_term_sign_dict((term_dict, sign_dict), test_forms)

    @classmethod
    def _parse_meshes(cls, term_dict):
        """"""
        meshes = list()
        for i in term_dict:
            for terms in term_dict[i]:
                for t in terms:
                    mesh = t.mesh
                    if mesh not in meshes:
                        meshes.append(mesh)
                    else:
                        pass

        num_meshes = len(meshes)
        assert num_meshes > 0, f"we need at least one mesh."

        if num_meshes == 1:
            mesh = meshes[0]  # config #1: one mesh found, then we are probably dealing with periodic manifold.
            return meshes, mesh  # RETURN config #1
        else:
            mesh_dim_dict = dict()
            for m in meshes:
                ndim = m.ndim
                if ndim not in mesh_dim_dict:
                    mesh_dim_dict[ndim] = list()
                else:
                    pass
                mesh_dim_dict[ndim].append(m)

            # below we analyze more than one mesh ---------------
            if len(mesh_dim_dict) == 2:   # meshes are of two dimensions.
                larger_ndim = max(mesh_dim_dict.keys())
                lower_ndim = min(mesh_dim_dict.keys())
                if lower_ndim == larger_ndim - 1 and \
                        len(mesh_dim_dict[larger_ndim]) == 1 and \
                        len(mesh_dim_dict[lower_ndim]) == 1:
                    mesh = mesh_dim_dict[larger_ndim][0]   # config #2: found one mesh, and its boundary mesh.
                    # This is the most common case.
                    boundary_mesh = mesh_dim_dict[lower_ndim][0]
                    assert mesh.manifold.boundary() == boundary_mesh.manifold, f"must be the case. Safety check."
                    return meshes, mesh  # RETURN config #2
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()

    def _consistence_checker(self):
        """We do consistence check here and parse properties like mesh and so on."""
        ts = list()
        for tf in self._test_forms:
            ts.append(tf.space)
        self._test_spaces = ts

        efs = set()
        for i in self._term_dict:   # ith equation
            for terms in self._term_dict[i]:
                for term in terms:
                    efs.update(term.elementary_forms)

        # below, lets sort the set according to the pure_lin_repr of the elementary forms and put it into a tuple.
        efs_dict = dict()
        efs_plr_list = list()
        for ef in efs:
            pure_lin_repr = ef._pure_lin_repr
            efs_dict[pure_lin_repr] = ef
            efs_plr_list.append(pure_lin_repr)
        efs_plr_list.sort()
        efs = [efs_dict[_] for _ in efs_plr_list]
        self._efs = tuple(efs)

    def _find_elementary_bc_forms(self):
        """find all elementary forms in bc terms; these forms will not be given as themselves."""
        e_bc_fs = set()
        e_bc_fs.update(
            self._find_elementary_forms_for_natural_bc()[0]
        )
        return e_bc_fs

    def _find_elementary_forms_for_natural_bc(self):
        """Even when bc is not defined. We can check it from the simple pattern of terms."""
        nbc_efs = set()
        other_efs = set()
        from src.config import _wf_term_default_simple_patterns as _simple_patterns
        for i in self._term_dict:   # ith equation
            for terms in self._term_dict[i]:
                for term in terms:
                    if term._simple_pattern == _simple_patterns['<tr star | tr >']:
                        two_element_forms = term.elementary_forms
                        assert len(two_element_forms) == 2, \
                            (f"term of pattern={_simple_patterns['<tr star | tr >']} "
                             f"must only have two elementary forms.")
                        for ef in two_element_forms:
                            if ef in self.test_forms:
                                pass
                            else:
                                nbc_efs.add(ef)

                    else:
                        other_efs.update(term.elementary_forms)

        return nbc_efs, other_efs

    def __repr__(self):
        """Customize the __repr__."""
        super_repr = super().__repr__().split('object')[1]
        if self.unknowns is None:
            unknown_sym_repr = '[...]'
        else:
            unknown_sym_repr = [f._sym_repr for f in self.unknowns]
        return f"<WeakFormulation of {unknown_sym_repr}" + super_repr

    def __getitem__(self, item):
        """"""
        assert item in self._indexing, \
            f"index: '{item}' is illegal, do 'print_representations(indexing=True)' " \
            f"to check indices of all terms."
        return self._indexing[item]

    @property
    def terms(self):
        """"""
        if self._terms is None:
            self._terms = _TermsOnly(self)
        return self._terms

    def __iter__(self):
        """"""
        for i in self._ind_dict:
            for lri in self._ind_dict[i]:
                for index in lri:
                    yield index

    def __len__(self):
        """"""
        return len(self._term_dict)

    def __contains__(self, item):
        return item in self._indexing

    def _parse_index(self, index):
        """"""
        ith_equation, k = index.split('-')
        ith_equation = int(ith_equation)
        k = int(k)
        left_terms = self._term_dict[ith_equation][0]
        number_left_terms = len(left_terms)
        if k < number_left_terms:
            l_o_r = 0  # left
            ith_term = k
        else:
            l_o_r = 1
            ith_term = k - number_left_terms
        return ith_equation, l_o_r, ith_term

    @property
    def elementary_forms(self):
        """Return a set of root forms that this equation involves."""
        return self._efs

    @property
    def unknowns(self):
        """The unknowns of this weak formulation"""
        return self._unknowns

    @unknowns.setter
    def unknowns(self, unknowns):
        """The unknowns of this weak formulation"""
        if self._unknowns is not None:
            raise Exception(f"unknowns exists; not allowed to change them.")
        if unknowns is None:
            self._unknowns = None
            return

        if len(self) == 1 and not isinstance(unknowns, (list, tuple)):
            unknowns = [unknowns, ]
        assert isinstance(unknowns, (list, tuple)), \
            f"please put unknowns in a list or tuple if there are more than 1 equations."
        assert len(unknowns) == len(self), \
            f"I have {len(self)} equations but receive {len(unknowns)} unknowns."

        for i, unknown in enumerate(unknowns):
            # noinspection PyUnresolvedReferences
            assert unknown.__class__.__name__ == 'Form' and unknown.is_root(), \
                f"{i}th variable is not a root form."
            assert unknown in self._efs, f"{i}th variable is not an elementary form."

        self._unknowns = unknowns

    @property
    def test_forms(self):
        return self._test_forms

    def _pr_pattern(self, indexing=True):
        """"""
        from src.config import RANK, MASTER_RANK
        if RANK != MASTER_RANK:
            return None
        else:
            pass

        pattern_text = ''

        for i in self._term_dict:
            terms = self._term_dict[i]
            signs = self._sign_dict[i]

            left_terms, right_terms = terms
            left_signs, right_signs = signs

            left_text = ''
            right_text = ''

            if len(left_terms) == 0:
                left_text = '0'
            else:
                for j, term in enumerate(left_terms):
                    sign = left_signs[j]
                    if j == 0 and sign == '+':
                        sign = ''
                    elif sign == '-':
                        sign = r'$-$'
                    else:
                        pass

                    pattern_str = term._simple_pattern
                    if pattern_str == '':
                        pattern_str = 'tbd.'

                    if indexing:
                        index = self._ind_dict[i][0][j]
                        pattern_str = r'$\underbrace{\text{' + pattern_str + r'}}_{' + \
                                      rf"{index}" + '}$'
                    else:
                        pass

                    left_text += sign + pattern_str

            if len(right_terms) == 0:
                right_text = '0'
            else:
                for j, term in enumerate(right_terms):
                    sign = right_signs[j]
                    if j == 0 and sign == '+':
                        sign = ''
                    elif sign == '-':
                        sign = r'$-$'
                    else:
                        pass

                    pattern_str = term._simple_pattern
                    if pattern_str == '':
                        pattern_str = 'tbd.'
                    if indexing:
                        index = self._ind_dict[i][1][j]
                        pattern_str = r'$\underbrace{\text{' + pattern_str + r'}}_{' + \
                                      rf"{index}" + '}$'
                    else:
                        pass

                    right_text += sign + pattern_str

            text_i = left_text + '=' + right_text

            if i < len(self._term_dict) - 1:
                text_i += '\n\n'

            pattern_text += text_i

        fig = plt.figure(figsize=(10, 5))
        plt.axis((0, 1, 0, 1))
        plt.axis('off')
        plt.text(0.05, 0.5, pattern_text, ha='left', va='center', size=15)
        from src.config import _setting, _pr_cache
        if _setting['pr_cache']:
            _pr_cache(fig, filename='weakFormulation_patterns')
        else:
            plt.tight_layout()
            plt.show(block=_setting['block'])
            plt.close()
        return fig

    def pr(self, indexing=True, patterns=False, saveto=None):
        """Print the representation of this weak formulation.

        Parameters
        ----------
        indexing : bool, optional
            Whether to show indices of the weak formulation terms. The default value is ``True``.
        patterns : bool, optional
            Whether to print the patterns of terms instead. The default value is ``False``.
        saveto : {None, str}, optional

        """
        from src.config import RANK, MASTER_RANK
        if RANK != MASTER_RANK:
            return None
        else:
            pass

        if patterns:
            return self._pr_pattern(indexing=indexing)
        else:
            pass

        seek_text = self._mesh.manifold._manifold_text()
        if self.unknowns is None:
            seek_text += r'for $\left('
            form_sr_list = list()
            space_sr_list = list()
            for ef in self._efs:
                if ef not in self._test_forms:
                    form_sr_list.append(rf' {ef._sym_repr}')
                    space_sr_list.append(rf"{ef.space._sym_repr}")
                else:
                    pass
            seek_text += ','.join(form_sr_list)
            seek_text += r'\right) \in '
            seek_text += r'\times '.join(space_sr_list)
            seek_text += '$, \n'
        else:
            given_text = r'for'
            for ef in self._efs:
                if ef not in self.unknowns and ef not in self._test_forms:
                    given_text += rf' ${ef._sym_repr} \in {ef.space._sym_repr}$, '
            if given_text == r'for':
                seek_text += r'seek $\left('
            else:
                seek_text += given_text + '\n'
                seek_text += r'seek $\left('
            form_sr_list = list()
            space_sr_list = list()
            for un in self.unknowns:
                form_sr_list.append(rf' {un._sym_repr}')
                space_sr_list.append(rf"{un.space._sym_repr}")
            seek_text += ','.join(form_sr_list)
            seek_text += r'\right) \in '
            seek_text += r'\times '.join(space_sr_list)
            seek_text += '$, such that\n'
        symbolic = ''
        number_equations = len(self._term_dict)
        for i in self._term_dict:
            for t, terms in enumerate(self._term_dict[i]):
                if len(terms) == 0:
                    symbolic += '0'
                else:

                    for j, term in enumerate(terms):
                        sign = self._sign_dict[i][t][j]
                        term = self._term_dict[i][t][j]

                        term_sym_repr = term._sym_repr

                        if indexing:
                            index = self._ind_dict[i][t][j].replace('-', r'\text{-}')
                            term_sym_repr = r'\underbrace{' + term_sym_repr + r'}_{' + \
                                rf"{index}" + '}'
                        else:
                            pass

                        if j == 0:
                            if sign == '+':
                                symbolic += term_sym_repr
                            elif sign == '-':
                                symbolic += '-' + term_sym_repr
                            else:
                                raise Exception()
                        else:
                            symbolic += ' ' + sign + ' ' + term_sym_repr

                if t == 0:
                    symbolic += ' &= '

            symbolic += r'\quad &&\forall ' + self._test_forms[i]._sym_repr + r'\in ' + \
                self._test_spaces[i]._sym_repr

            if i < number_equations - 1:
                symbolic += r',\\'
            else:
                symbolic += '.'

        symbolic = r"$\left\lbrace\begin{aligned}" + symbolic + r"\end{aligned}\right.$"
        if self._bc is None or len(self.bc) == 0:
            bc_text = ''
        else:
            bc_text = self.bc._bc_text()

        if indexing:
            height = 2 * len(self._term_dict)
        else:
            height = 1.5 * len(self._term_dict)
        if height > 6:
            height = 6
        figsize = (10, height)

        fig = plt.figure(figsize=figsize)
        plt.axis((0, 1, 0, 1))
        plt.axis('off')
        plt.text(0.05, 0.5, seek_text + symbolic + bc_text, ha='left', va='center', size=15)

        if saveto is not None:
            plt.savefig(saveto, bbox_inches='tight', dpi=200)
            return None
        else:
            from src.config import _setting, _pr_cache
            if _setting['pr_cache']:
                _pr_cache(fig, filename='weakFormulation')
            else:
                plt.tight_layout()
                plt.show(block=_setting['block'])
                plt.close()
            return fig

    @property
    def derive(self):
        """A wrapper all possible derivations to the weak formulation."""
        if self._derive is None:
            self._derive = WfDerive(self)
        return self._derive

    @property
    def bc(self):
        """The boundary condition of the weak formulation."""
        if self._bc is None:
            self._bc = BoundaryCondition(self._mesh)
        return self._bc

    @property
    def td(self):
        """Temporal discretization of the weak formulation."""
        return TemporalDiscretization(self)

    def ap(self):
        """"""
        return AlgebraicProxy(self)

    def mp(self):
        """Generate a matrix proxy for the weak formulation.

        Returns
        -------
        mp : :class:`src.wf.mp.main.MatrixProxy`
        """
        return MatrixProxy(self)

    def _pr_temporal_advancing(self, ts, time_instant_hierarchy):
        """This method should be called from somewhere else."""
        from src.config import RANK, MASTER_RANK
        if RANK != MASTER_RANK:
            return
        else:
            pass

        elementary_forms = self.elementary_forms
        assert self.unknowns is not None, f"set unknowns before plotting temporal advancing."
        known_forms = list()
        ati_collection = list()
        ati_keys = list()
        non_test_forms = list()

        for ef in elementary_forms:
            if ef._pAti_form['base_form'] is None:
                assert ef._pure_lin_repr[-len(_pde_test_form_lin_repr):] == _pde_test_form_lin_repr, \
                    f"form {ef} is not specified at an (abstract) time instant"
            else:
                # first we classify them into known and unknowns.
                if ef in self.unknowns:
                    pass
                else:
                    known_forms.append(ef)
                non_test_forms.append(ef)
                # check ts
                ats = ef._pAti_form['ats']
                assert ats._object is not None, f"elementary form {ef} only has an abstract time sequence."
                assert ats._object is ts, f"time sequence of elementary form {ef} does not match the input ts."
                # then lets collect all abstract time instants.
                ati = ef._pAti_form['ati']
                ati_collection.append(ati)
                ati_keys.extend(ati._kwarg_keys)

        ati_keys = list(set(ati_keys))  # remove repeated akt keys
        ati_keys.sort()

        # ---- found special elementary forms ------------------
        nbc_efs, nbc_other_efs = self._find_elementary_forms_for_natural_bc()  # 1

        # group forms according to their base form ------------
        ntf_group = dict()
        for ntf in non_test_forms:
            bf = ntf._pAti_form['base_form']
            bf_plr = bf._pure_lin_repr
            if bf_plr in ntf_group:
                pass
            else:
                ntf_group[bf_plr] = list()
            ntf_group[bf_plr].append(ntf)

        from src.time_sequence import ConstantTimeSequence
        evaluate_sym = _form_evaluate_at_repr_setting['sym']
        if ts.__class__ is ConstantTimeSequence:
            vertical_length = (ts._t_max - ts._t_0) * 0.02
            if len(ati_keys) == 1:
                key = ati_keys[0]
                major_nodes = time_instant_hierarchy[0]
                num_plots = len(major_nodes[1:])
                for k in range(1, num_plots+1):
                    fig, ax = plt.subplots(figsize=(12, 6))
                    i = 3
                    j = - 5.5
                    for bf_plr in ntf_group:
                        tfs = ntf_group[bf_plr]
                        bf = tfs[0]._pAti_form['base_form']

                        abstract_forms = list()
                        else_forms = list()
                        for tf in tfs:
                            ati = tf._pAti_form['ati']
                            if key in ati._kwarg_keys:
                                abstract_forms.append(tf)
                            else:
                                assert ati.ati._kwarg_keys == [], f"must be since len(ati_keys) == 1"
                                else_forms.append(tf)

                        bf_sym_repr = bf._sym_repr
                        if len(abstract_forms) > 0:
                            y_location = i * vertical_length
                            i += 2.75
                            plt.text(  # plot its base form at the most left place
                                -ts._dt, y_location,
                                f'${bf_sym_repr}$',
                                va='center', ha='right', fontsize=15)

                            for af in abstract_forms:
                                ati = af._pAti_form['ati']
                                time = ati(**{f'{key}': k})()
                                time_instance_str = ati._k.replace(key, str(k))
                                time_instance = str(eval(time_instance_str))
                                sym_repr = (
                                        evaluate_sym[0] +
                                        bf_sym_repr +
                                        evaluate_sym[1] +
                                        time_instance +
                                        evaluate_sym[2]
                                )
                                if af in self.unknowns:
                                    color = 'red'
                                    text = f"${sym_repr}$"
                                else:
                                    assert af in known_forms, f'A safety check, must be!'
                                    if af in nbc_other_efs:
                                        text = f"${sym_repr}$"
                                        color = 'blue'
                                    else:
                                        color = None
                                        text = None

                                if color is None:
                                    pass
                                else:
                                    plt.text(
                                        time, y_location, text,
                                        c=color,
                                        va='center', ha='center', fontsize=15,
                                    )

                                if af in nbc_efs:  # if we found a natural bc, we plot it additionally.
                                    plt.text(
                                        time, y_location,
                                        r"$\underbrace{\mathrm{tr}\star " + f"{sym_repr}" + r"}_{\partial}$",
                                        c='gray',
                                        va='center', ha='center', fontsize=15,
                                    )
                                else:
                                    pass

                        else:
                            pass

                        if len(else_forms) > 0:
                            # these forms will be plotted at the lower part.
                            ng_y_location = j * vertical_length
                            j -= 2.5
                            for pf in else_forms:  # pf stands for particular form whose time is not abstract.
                                ati = pf._pAti_form['ati']
                                time = ati()()
                                color = 'k'
                                plt.text(
                                    time, ng_y_location, f"${pf._sym_repr}$",
                                    c=color,
                                    va='center', ha='center', fontsize=15,
                                )
                        else:
                            pass

                    i += 2.75
                    y_location = i * vertical_length
                    plt.text(
                        ts._dt * k, y_location,
                        f'${key}={k}$',
                        c='k',
                        va='center', ha='center', fontsize=15,
                    )
                    plt.plot(   # ending vertical line
                        [ts._dt * k, ts._dt * k], [j * vertical_length, y_location],
                        '--',
                        color='pink',
                        linewidth=0.8,
                    )

                    ax.set_aspect('equal')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    plt.tick_params(left=False,
                                    right=False,
                                    labelleft=False,
                                    labelbottom=False,
                                    bottom=False)
                    plt.plot(  # intermediate major time instants
                        major_nodes[1:], 0*major_nodes[1:],
                        '-s',
                        color='k',
                        linewidth=1.2,
                                         )
                    plt.plot(
                        [ts._t_0-ts._dt, ts._t_max+ts._dt], [0, 0],
                        '-',
                        color='k',
                        linewidth=1.2,
                    )
                    right_end = ts._t_max+ts._dt
                    plt.plot(   # start vertical line
                        [ts._t_0, ts._t_0], [j * vertical_length, y_location],
                        '--',
                        color='lightgreen',
                        linewidth=0.8,
                    )
                    plt.plot(   # start point
                        [ts._t_0, ts._t_0], [-vertical_length, vertical_length],
                        color='darkgreen',
                        linewidth=1.8,
                    )
                    plt.plot(   # ending vertical line
                        [ts._t_max, ts._t_max], [j * vertical_length, y_location],
                        '--',
                        color='peru',
                        linewidth=0.8,
                    )
                    plt.plot(  # ending triangle
                        [right_end-vertical_length, right_end, right_end-vertical_length],
                        [vertical_length, 0, -vertical_length],
                        color='k',
                        linewidth=1,
                    )
                    for _k_, major_node in enumerate(major_nodes):
                        if _k_ == 0:
                            plt.text(
                                major_node,
                                -3*vertical_length,
                                f'$t_{_k_}=%.1f$' % ts._t_0,
                                c='darkgreen',
                                ha='center', va='center',
                                fontsize=15
                            )
                        elif _k_ == len(major_nodes) - 1:
                            plt.text(
                                major_node,
                                -3*vertical_length,
                                f'$t_{_k_}=%.1f$' % ts._t_max,
                                c='brown',
                                ha='center', va='center',
                                fontsize=15
                            )
                        else:
                            plt.text(
                                major_node,
                                -3*vertical_length,
                                f'$t_{_k_}$',
                                ha='center', va='center',
                                fontsize=15
                            )

                    if 1 in time_instant_hierarchy:
                        minor_nodes = time_instant_hierarchy[1]
                        plt.scatter(
                            minor_nodes, 0*minor_nodes, marker='x', color='gray'
                        )

                    # ----- save -----------------------------------------------------
                    plt.tight_layout()
                    from src.config import _setting, _pr_cache
                    if _setting['pr_cache']:
                        _pr_cache(fig, filename='weakFormulationTimeAdvancing')
                    else:
                        plt.show(block=_setting['block'])
            else:
                raise NotImplementedError(
                    f"'_pr_temporal_advancing' not implemented for multiple ati keys: {ati_keys}"
                )
        else:
            raise NotImplementedError(f"cannot plot temporal advancing for wf of time sequence {ts.__class__}")


class _TermsOnly(Frozen):
    """The collection of wf terms only."""
    def __init__(self, wf):
        self._wf = wf
        self._freeze()

    def __getitem__(self, item):
        sign, term = self._wf[item]
        return term


if __name__ == '__main__':
    # python src/wf/matplot.py

    import __init__ as ph
    # import phlib as ph
    # ph.config.set_embedding_space_dim(3)
    manifold = ph.manifold(3)
    mesh = ph.mesh(manifold)

    ph.space.set_mesh(mesh)
    O0 = ph.space.new('Lambda', 0)
    O1 = ph.space.new('Lambda', 1)
    O2 = ph.space.new('Lambda', 2)
    O3 = ph.space.new('Lambda', 3)

    # ph.list_spaces()
    # ph.list_meshes()

    w = O1.make_form(r'\omega^1', "vorticity1")
    u = O2.make_form(r'u^2', "velocity2")
    f = O2.make_form(r'f^2', "body-force")
    P = O3.make_form(r'P^3', "total-pressure3")

    wXu = w.wedge(ph.Hodge(u))
    dsP = ph.codifferential(P)
    dsu = ph.codifferential(u)
    du = ph.d(u)
    du_dt = ph.time_derivative(u)

    # ph.list_forms(globals())

    exp = [
        'du_dt + wXu - dsP = f',
        'w = dsu',
        'du = 0',
    ]
    pde = ph.pde(exp, globals())
    pde.unknowns = [u, w, P]
    # pde.pr(indexing=True)

    wf = pde.test_with([O2, O1, O3], sym_repr=[r'v^2', r'w^1', r'q^3'])

    wf = wf.derive.integration_by_parts('0-2')
    wf = wf.derive.integration_by_parts('1-1')

    wf = wf.derive.rearrange(
        {
            0: '0, 1, 2 = 4, 3',
            1: '1, 0 = 2',
            2: ' = 0',
        }
    )

    wf.pr()

    # i = 0
    # terms = wf._term_dict[i]
    # signs = wf._sign_dict[i]
    #
    # ode_i = ph.ode(terms_and_signs=[terms, signs])
    # ode_i.print_representations()
    # ph.list_forms()

    # td = wf.td
    # ap = wf.ap
