# -*- coding: utf-8 -*-
# noinspection PyUnresolvedReferences
r"""

.. _docs-form:

Form
====

A form is simply an element of a space. Thus, it is logical to define a form through a space. To do that,
just call the ``make_form`` method of the space instance (see :meth:`src.spaces.base.SpaceBase.make_form`).
The following code makes an outer oriented 2-form, ``a``, in space ``Out2`` and
an outer-oriented 1-form, ``b``, in space ``Out1``.

>>> a = Out2.make_form(r'\tilde{\alpha}', 'variable1')
>>> b = Out1.make_form(r'\tilde{\beta}', 'variable2')

The arguments are their symbolic representations (``r'\tilde{\alpha}'``, ``r'\tilde{\beta}'``) and
linguistic representations (``'variable1'``, ``'variable2'``), represectively. To list all defined forms, do

>>> ph.list_forms()
<Figure size ...

If you have turned off the *py cache* , see :ref:`docs-presetting-set`, a figure should have popped out. Otherwise,
it is saved to ``./__phcache__/Pr_current/``. A form is an instance of :class:`Form`.

    .. autoclass:: src.form.main.Form
        :members: mesh, manifold, orientation, space, wedge, codifferential, exterior_derivative, cross_product,
            time_derivative, degree

The forms ``a`` and ``b`` are root forms since they are directly defined through ``make_form`` method.
With these elementary forms, it is possible to build more complicated non-root forms through operators.
Implemented operators are

.. admonition:: Implemented operators for forms

    +----------------------+--------------------------------------+--------------------------------------+
    | **operator**         |**symbolic representation**           |    **usage**                         |
    +----------------------+--------------------------------------+--------------------------------------+
    | exterior derivative  | :math:`\mathrm{d}`                   | ``a.exterior_derivative()``          |
    |                      |                                      | or ``ph.exteror_derivative(a)``      |
    |                      |                                      | or ``ph.d(a)``                       |
    +----------------------+--------------------------------------+--------------------------------------+
    | codifferential       | :math:`\mathrm{d}^\ast`              | ``a.codifferential()``               |
    |                      |                                      | or ``ph.codifferential(a)``          |
    +----------------------+--------------------------------------+--------------------------------------+
    | time derivative      | :math:`\frac{\partial}{\partial t}`  | ``a.time_derivative()``              |
    |                      |                                      | or ``ph.time_derivative(a)``         |
    +----------------------+--------------------------------------+--------------------------------------+
    | wedge product        | :math:`\wedge`                       | Given two forms,                     |
    |                      |                                      | ``a``: :math:`\alpha` and            |
    |                      |                                      | ``b``: :math:`\beta`,                |
    |                      |                                      | The wedge product between            |
    |                      |                                      | them, i.e.                           |
    |                      |                                      | :math:`\alpha\wedge\beta`,           |
    |                      |                                      | is ``a.wedge(b)`` or                 |
    |                      |                                      | ``ph.wedge(a, b)``.                  |
    |                      |                                      |                                      |
    |                      |                                      |                                      |
    +----------------------+--------------------------------------+--------------------------------------+
    | inner product        | :math:`\left(\cdot,\cdot\right)`     | Given two forms,                     |
    |                      |                                      | ``a``: :math:`\alpha` and            |
    |                      |                                      | ``b``: :math:`\beta`,                |
    |                      |                                      | The inner product between            |
    |                      |                                      | them, i.e.                           |
    |                      |                                      | :math:`\left(\alpha,\beta\right)`,   |
    |                      |                                      | is ``ph.inner(a,b)``.                |
    |                      |                                      |                                      |
    +----------------------+--------------------------------------+--------------------------------------+
    | Hodge                | :math:`\star`                        | ``ph.Hodge(a)``                      |
    +----------------------+--------------------------------------+--------------------------------------+
    | trace                | :math:`\mathrm{tr}`                  | ``ph.trace(a)``                      |
    +----------------------+--------------------------------------+--------------------------------------+

    ..
        | cross product        | :math:`\times`                       | Given two forms,                     |
        |                      |                                      | ``a``: :math:`\alpha` and            |
        |                      |                                      | ``b``: :math:`\beta`,                |
        |                      |                                      | The cross product between            |
        |                      |                                      | them, i.e.                           |
        |                      |                                      | :math:`\alpha\times\beta`,           |
        |                      |                                      | is ``a.cross_product(b)``.           |
        |                      |                                      |                                      |
        +----------------------+--------------------------------------+--------------------------------------+


For example,

>>> da_dt = a.time_derivative()
>>> db_dt = b.time_derivative()
>>> cd_a = a.codifferential()
>>> d_b = b.exterior_derivative()

This generates four more forms, ``da_dt``, ``db_dt``, ``cd_a`` and ``d_b``, which are

- time derivative of ``a``
- time derivative of ``b``
- codifferential of ``a``
- exterior derivative of ``b``

respectively. These non-root forms will appear in the form list if you do

>>> ph.list_forms()
<Figure size ...

All these forms, both root and non-root ones, are the ingredients for making partial differential equaitons (DPE)
which is introduced in the next section.

"""
from typing import Dict
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "text.latex.preamble": r"\usepackage{amsmath}"
})
matplotlib.use('TkAgg')

from phyem.tools.frozen import Frozen
from phyem.src.config import _global_lin_repr_setting
from phyem.src.config import _parse_lin_repr
from phyem.src.form.operators import wedge, time_derivative, d, codifferential, cross_product, tensor_product
from phyem.src.form.operators import Cross_Product, CrossProduct, crossProduct
from phyem.src.form.operators import convect, multi
from phyem.src.form.operators import _project_to
from phyem.src.config import _check_sym_repr
from phyem.src.form.parameters import constant_scalar
from phyem.src.config import _global_operator_lin_repr_setting
from phyem.src.config import _global_operator_sym_repr_setting
from phyem.src.config import _form_evaluate_at_repr_setting
from phyem.src.spaces.main import _default_space_degree_repr
from phyem.src.spaces.main import _degree_str_maker


_global_forms = dict()   # cache keys are id
_global_root_forms_lin_dict = dict()   # keys are root form lin_repr
_global_form_variables = {
    'update_cache': True,   # the global switcher  ---------- (1)
}


def _clear_forms():
    """"""
    for key in list(_global_forms.keys()):
        del _global_forms[key]
    for key in list(_global_root_forms_lin_dict.keys()):
        del _global_root_forms_lin_dict[key]


from phyem.src.form.ap import _parse_root_form_ap


class Form(Frozen):
    """The form class."""

    def __init__(
            self, space,
            sym_repr, lin_repr,
            is_root,
            update_cache=True,   # the local switcher  ------------ (1)
    ):
        if is_root is None:  # we will parse is_root from lin_repr
            assert isinstance(lin_repr, str) and len(lin_repr) > 0, f"lin_repr={lin_repr} illegal."
            is_root, lin_repr = self._parse_is_root(lin_repr)
        else:
            pass
        assert isinstance(is_root, bool), f"is_root must be bool."
        self._space = space

        if is_root:  # we check the `sym_repr` only for root forms.
            lin_repr, self._pure_lin_repr = _parse_lin_repr('form', lin_repr)
            for form_id in _global_forms:
                form = _global_forms[form_id]
                assert sym_repr != form._sym_repr, \
                    f"root form symbolic representation={sym_repr} is taken. Pls use another one."
                assert lin_repr != form._lin_repr, \
                    f"root form linguistic representation={lin_repr} is taken. Pls use another one."
        else:
            self._pure_lin_repr = None

        sym_repr = _check_sym_repr(sym_repr)
        self._sym_repr = sym_repr
        self._lin_repr = lin_repr
        self._is_root = is_root
        self._efs = None   # elementary elements
        self._orientation = space.orientation
        if update_cache:
            if _global_form_variables['update_cache']:  # cache it
                _global_forms[id(self)] = self
                if self._is_root:
                    _global_root_forms_lin_dict[self._lin_repr] = self
                else:
                    pass
            else:
                pass
        else:
            pass
        self._pAti_form: Dict = {
            'base_form': None,
            'ats': None,
            'ati': None
        }
        self._ats_forms = dict()   # the abstract ats forms based on this form.
        self._degree = None
        self._ap = None
        self._dual_representation = False
        self._freeze()

    def is_dual_representation(self):
        return self._dual_representation

    def set_dual_representation(self, _bool):
        assert isinstance(_bool, bool)
        self._dual_representation = _bool

    # noinspection PyBroadException
    @staticmethod
    def _parse_is_root(lin_repr):
        """Study is_root through lin_repr."""
        try:
            _parse_lin_repr('form', lin_repr)

        except Exception:
            pass

        else:
            return True, lin_repr

        start, end = _global_lin_repr_setting['form']

        if lin_repr[:len(start)] == start and lin_repr[-len(end):] == end:

            try:
                _parse_lin_repr('form', lin_repr[len(start):-len(end)])

            except Exception:
                return False, lin_repr

            else:
                return True, lin_repr[len(start):-len(end)]

        else:
            return False, lin_repr

    def pr(self, figsize=(12, 6)):
        """Print this form with matplotlib and latex."""
        from phyem.src.config import RANK, MASTER_RANK
        if RANK != MASTER_RANK:
            return None
        else:
            my_id = r'\texttt{' + str(id(self)) + '}'
            if self._pAti_form['base_form'] is None:
                pti_text = ''
            else:
                base_form, ats, ati = self._pAti_form['base_form'], self._pAti_form['ats'], self._pAti_form['ati']
                pti_text = rf"\\(${base_form._sym_repr}$ at abstract time instant ${ati._sym_repr}$"
            space_text = f'spaces: ${self.space._sym_repr}$'
            space_text += rf"\ \ \ \ on ({self.mesh._lin_repr})"
            fig = plt.figure(figsize=figsize)
            plt.axis((0, 1, 0, 5))
            plt.text(0, 4.5, f'form id: {my_id}', ha='left', va='center', size=15)
            plt.text(0, 3.5, space_text, ha='left', va='center', size=15)
            plt.text(0, 2.5,
                     rf'\noindent symbolic: ' + f"${self._sym_repr}$" + pti_text,
                     ha='left', va='center', size=15)
            plt.text(0, 1.5, 'linguistic: ' + self._lin_repr, ha='left', va='center', size=15)
            root_text = rf'is_root: {self.is_root()}'
            plt.text(0, 0.5, root_text, ha='left', va='center', size=15)
            plt.axis('off')
            from phyem.src.config import _setting
            plt.show(block=_setting['block'])
            return fig

    def __repr__(self):
        """"""
        super_repr = super().__repr__().split('object')[-1]
        return '<Form ' + self._sym_repr + super_repr  # this will be unique.

    @property
    def elementary_forms(self):
        """parse the elementary_forms from the linguistic representation only. A texting solution only!"""
        if self._efs is None:
            efs = list()
            for root_lin_repr in _global_root_forms_lin_dict:
                if root_lin_repr in self._lin_repr:
                    efs.append(_global_root_forms_lin_dict[root_lin_repr])
            self._efs = set(efs)
        return self._efs

    @property
    def degree(self):
        """This form is in the space of particular finite dimensional ``degree``."""
        assert self._degree is not None, f"degree of form {self} is empty, set it firstly."
        return self._degree

    @degree.setter
    def degree(self, _degree):
        """Limit this form to a particular finite dimensional space of degree ``_degree``."""
        assert isinstance(_degree, (int, float, list, tuple)), f"Can only use int, float, list or tuple for the degree."

        for _lin_repr in _global_root_forms_lin_dict:
            root_form = _global_root_forms_lin_dict[_lin_repr]
            if root_form._pAti_form['base_form'] is self:
                root_form._degree = _degree

        self.space.finite.specify_form(self, _degree)

    def ap(self, sym_repr=None):
        """Algebraic proxy."""
        if self._ap is None:
            if self.is_root():
                self._ap = _parse_root_form_ap(self, sym_repr)
            else:
                raise Exception("None root form has no symbolic representation, do not try to access.")
        else:
            assert sym_repr is None, f"form {self} already have an algebraic proxy, change its symbolic " \
                                     f"representation is not allowed (cause it may not be safe)."
        return self._ap

    def _ap_shape(self):
        """ap shape."""
        return self.space._sym_repr + _default_space_degree_repr + _degree_str_maker(self._degree)

    @property
    def orientation(self):
        """My orientation."""
        return self._orientation

    def is_root(self):
        """Return True this form is a root form."""
        return self._is_root

    @property
    def space(self):
        """The space this form is in."""
        return self._space

    @property
    def mesh(self):
        """The mesh this form is on."""
        return self.space.mesh

    @property
    def manifold(self):
        """The manifold this form is on."""
        return self.mesh.manifold

    # ------------------------------------------------------------------------------------

    def representing(self):
        r"""What does this form representing?

        So, return m, n, and scalar or vector or tensor or somthing similar!

        """
        m = self.space.m
        n = self.space.n
        indicator = self.space.indicator

        if indicator == 'Lambda':
            k = self.space.k
            if m == n == 2:
                if k == 1:
                    v_type = 'vector'
                elif k == 0 or k == 2:
                    v_type = 'scalar'
                else:
                    raise Exception()
            elif m == n == 3:
                if k == 1 or k == 2:
                    v_type = 'vector'
                elif k == 0 or k == 3:
                    v_type = 'scalar'
                else:
                    raise Exception()
            else:
                raise NotImplementedError()

            return ['Lambda', m, n, v_type]

        else:
            raise NotImplementedError()

    # --------------- OPERATORS ----------------------------------------------------------

    def wedge(self, other):
        r"""The wedge, :math:`\wedge`, between this form and another form."""
        return wedge(self, other)

    def time_derivative(self, degree=1):
        r"""The time derivative, :math:`\dfrac{\partial}{\partial t}`, of this form."""
        return time_derivative(self, degree=degree)

    def exterior_derivative(self):
        r"""The exterior derivative, :math:`\mathrm{d}`, of this form."""
        return d(self)

    def d(self):
        r""""""
        return self.exterior_derivative()

    def codifferential(self):
        r"""The codifferential, :math:`\mathrm{d}^\ast`, of this form."""
        return codifferential(self)

    def x(self, other, form_space=None):
        r"""

        Parameters
        ----------
        other
        form_space :
            i.e. the output space. We want to put the output to this space.

        Returns
        -------

        """
        try:
            output1 = self.cross_product(other)
        except NotImplementedError:
            output1 = None

        try:
            output2 = self.Cross_Product(other)
        except NotImplementedError:
            output2 = None

        try:
            output3 = self.CrossProduct(other)
        except NotImplementedError:
            output3 = None

        try:
            output4 = self.crossProduct(other)
        except NotImplementedError:
            output4 = None

        outputs = [output1, output2, output3, output4]
        if all([_ is None for _ in outputs]):
            raise NotImplementedError(
                f"implement a cross-product somewhere!")
        else:

            if form_space.__class__ is self.__class__:
                target_space = form_space.space
            else:
                raise NotImplementedError(f"which space?")

            CANNOT_implement_in = []
            if output1 is not None:
                if output1.space is target_space:
                    return output1
                else:
                    CANNOT_implement_in.append('cross_product')
            else:
                pass

            if output2 is not None:
                if output2.space is target_space:
                    return output2
                else:
                    CANNOT_implement_in.append('Cross_Product')
            else:
                pass

            if output3 is not None:
                if output3.space is target_space:
                    return output3
                else:
                    CANNOT_implement_in.append('CrossProduct')
            else:
                pass

            if output4 is not None:
                if output4.space is target_space:
                    return output4
                else:
                    CANNOT_implement_in.append('crossProduct')
            else:
                pass

            if len(CANNOT_implement_in) == 4:
                raise Exception()
            else:
                cp = f"{self.space} x {other.space}"
                if 'cross_product' not in CANNOT_implement_in:
                    raise NotImplementedError(f"implement this cross-product: {cp} -> {target_space} in cross_product")
                elif 'Cross_Product' not in CANNOT_implement_in:
                    raise NotImplementedError(f"implement this cross-product: {cp} -> {target_space} in Cross_Product")
                elif 'CrossProduct' not in CANNOT_implement_in:
                    raise NotImplementedError(f"implement this cross-product: {cp} -> {target_space} in CrossProduct")
                elif 'crossProduct' not in CANNOT_implement_in:
                    raise NotImplementedError(f"implement this cross-product: {cp} -> {target_space} in crossProduct")
                else:
                    raise Exception()

    def cross_product(self, other):
        r"""The cross product, :math:`\times`, between this form and another form."""
        return cross_product(self, other)

    def Cross_Product(self, other):
        r"""Another branch of cross-product."""
        return Cross_Product(self, other)

    def CrossProduct(self, other):
        r"""Another branch of cross-product."""
        return CrossProduct(self, other)

    def crossProduct(self, other):
        r"""Another branch of cross-product."""
        return crossProduct(self, other)

    def convect(self, other):
        """Let self be u, other be w, we compute u dot(grad(w))."""
        return convect(self, other)

    def tensor_product(self, other):
        """"""
        return tensor_product(self, other)

    def project_to(self, to_space):
        return _project_to(self, to_space)

    def multi(self, other, output_space):
        r"""Do f1 f2 -> a form in output space. For example, c = ab where a, b are both scalars, or D = a B where
        a is scalar and BD are vectors.
        """
        if output_space.__class__ is self.__class__:
            output_space = output_space.space
        else:
            raise NotImplementedError()
        return multi(self, other, output_space)

    def __neg__(self):
        """- self"""
        raise NotImplementedError()

    def __add__(self, other):
        """self + other"""
        if other.__class__.__name__ == 'Form':
            assert other.mesh == self.mesh, f"mesh does not match."
            assert self.orientation == other.orientation
            assert self.space == other.space
            self_lr = self._lin_repr
            self_sr = self._sym_repr
            other_lr = other._lin_repr
            other_sr = other._sym_repr

            operator_lin = _global_operator_lin_repr_setting['plus']
            operator_sym = _global_operator_sym_repr_setting['plus']

            lin_repr = self_lr + operator_lin + other_lr
            sym_repr = self_sr + operator_sym + other_sr

            f = Form(
                self.space,        # space
                sym_repr,          # symbolic representation
                lin_repr,          # linguistic representation
                False,      # must not be a root-form anymore.
            )
            return f

        else:
            raise NotImplementedError(f"{other}")

    def __sub__(self, other):
        """self-other"""
        if other.__class__.__name__ == 'Form':
            assert other.mesh == self.mesh, f"mesh does not match."
            assert self.orientation == other.orientation
            assert self.space == other.space
            self_lr = self._lin_repr
            self_sr = self._sym_repr
            other_lr = other._lin_repr
            other_sr = other._sym_repr

            operator_lin = _global_operator_lin_repr_setting['minus']
            operator_sym = _global_operator_sym_repr_setting['minus']

            lin_repr = self_lr + operator_lin + other_lr
            sym_repr = self_sr + operator_sym + other_sr

            f = Form(
                self.space,         # space
                sym_repr,           # symbolic representation
                lin_repr,           # linguistic representation
                False,       # must not be a root-form anymore.
            )
            return f

        else:
            raise NotImplementedError(f"{other}")

    def __mul__(self, other):
        """self * other"""
        raise NotImplementedError()

    def __rmul__(self, other):
        """other * self"""
        if isinstance(other, (int, tuple)):
            cs = constant_scalar(other)
            return cs * self
        elif other.__class__.__name__ == 'ConstantScalar0Form':
            operator_lin = _global_operator_lin_repr_setting['multiply']
            lr = self._lin_repr
            sr = self._sym_repr
            cs = other
            if self.is_root():
                lr = cs._lin_repr + operator_lin + lr
                sr = cs._sym_repr + sr
            else:
                if cs.is_root():
                    lr = cs._lin_repr + operator_lin + r'\{' + lr + r'\}'
                    sr = cs._sym_repr + r'\left(' + sr + r'\right)'
                else:
                    lr = r'\{' + cs._lin_repr + r'\}' + operator_lin + r'\{' + lr + r'\}'
                    sr = r'\left(' + cs._sym_repr + r'\right)' + r'\left(' + sr + r'\right)'

            f = Form(
                self.space,         # space
                sr,                 # symbolic representation
                lr,                 # linguistic representation
                False,       # not a root-form anymore.
            )
            return f
        else:
            raise NotImplementedError()

    def __truediv__(self, other):
        """self / other"""
        operator_lin = _global_operator_lin_repr_setting['division']
        operator_sym = _global_operator_sym_repr_setting['division']
        if isinstance(other, (int, tuple)):
            cs = constant_scalar(other)
            return self / cs

        elif other.__class__.__name__ == 'AbstractTimeInterval':
            ati = other
            return self / ati._as_scalar()

        elif other.__class__.__name__ == 'ConstantScalar0Form':
            lr = self._lin_repr
            sr = self._sym_repr
            cs = other
            if self.is_root():
                lr = lr + operator_lin + cs._lin_repr
            else:
                lr = r'\{' + lr + r'\}' + operator_lin + cs._lin_repr
            sr = operator_sym[0] + sr + operator_sym[1] + cs._sym_repr + operator_sym[2]
            f = Form(
                self.space,         # space
                sr,                 # symbolic representation
                lr,                 # linguistic representation
                False,       # not a root-form anymore.
            )
            return f

        else:
            raise NotImplementedError(f"form divided by <{other.__class__.__name__}> is not implemented.")

    def _evaluate_at(self, other):
        """evaluate_at"""
        from phyem.src.time_sequence import AbstractTimeInstant
        from phyem.src.time_sequence import _global_abstract_time_sequence

        if isinstance(other, str) and len(_global_abstract_time_sequence) == 1:
            # when there is only one abstract time sequence at behind, we can
            # access its abstract time instant by str directly.
            the_only_ats_lin_repr = list(_global_abstract_time_sequence.keys())[0]
            the_only_ats = _global_abstract_time_sequence[the_only_ats_lin_repr]
            other = the_only_ats[other]
        else:
            pass

        if other.__class__ is AbstractTimeInstant:
            ati = other
            assert self.is_root(), f"Can only evaluate a root form at an abstract time instant."
            sym_repr = self._sym_repr
            lin_repr = self._pure_lin_repr
            s = _form_evaluate_at_repr_setting['sym']
            sym_repr = s[0] + sym_repr + s[1] + ati.k + s[2]
            lin_repr += _form_evaluate_at_repr_setting['lin'] + ati._pure_lin_repr

            if lin_repr in self._ats_forms:   # we must cache it, this is very important.
                pass
            else:
                ftk = Form(
                    self._space,
                    sym_repr, lin_repr,
                    self.is_root(),  # must be True.
                )
                ftk._pAti_form['base_form'] = self
                ftk._pAti_form['ats'] = ati.time_sequence
                ftk._pAti_form['ati'] = ati

                ftk.set_dual_representation(self._dual_representation)

                self._ats_forms[lin_repr] = ftk

            return self._ats_forms[lin_repr]

        else:
            raise NotImplementedError(f"Cannot evaluate {self} at {other}.")

    def __matmul__(self, other):
        """self @ other"""
        return self._evaluate_at(other)

    def replace(self, f, by, which='all'):
        """replace `which` `f` by `by`."""
        assert by.space == f.space, f"spaces do not match."
        if f._lin_repr not in self._lin_repr:
            return self
        elif self._lin_repr == f._lin_repr:
            return by

        else:
            if which == 'all':
                lin_repr = self._lin_repr.replace(f._lin_repr, by._lin_repr)
                sym_repr = self._sym_repr.replace(f._sym_repr, by._sym_repr)

            elif isinstance(which, (list, tuple)) and all([isinstance(_, int) and _ >= 0 for _ in which]):
                # use 1, or [0,1] to indicate which targets to be replaced.
                lin_list = self._lin_repr.split(f._lin_repr)
                sym_list = self._sym_repr.split(f._sym_repr)
                assert len(lin_list) == len(sym_list), f'must be!'

                if len(lin_list) == 1:  # no replace target found!
                    raise Exception('found no target to replace.')
                else:
                    pass
                amount_places = len(lin_list) - 1
                to_join_lin = ['' for _ in range(amount_places)]
                to_join_sym = ['' for _ in range(amount_places)]
                for i in which:
                    assert i < amount_places, (f"cannot deal with `which={which}` for form replace, "
                                               f"not that many targets.)")
                    to_join_lin[i] = by._lin_repr
                    to_join_sym[i] = by._sym_repr

                for i, s in enumerate(to_join_lin):
                    if s == '':
                        to_join_lin[i] = f._lin_repr
                    else:
                        pass

                for i, s in enumerate(to_join_sym):
                    if s == '':
                        to_join_sym[i] = f._sym_repr
                    else:
                        pass

                final_lin_repr = ''
                final_sym_repr = ''
                for i in range(amount_places):
                    final_lin_repr += lin_list[i] + to_join_lin[i]
                    final_sym_repr += sym_list[i] + to_join_sym[i]
                final_lin_repr += lin_list[-1]
                final_sym_repr += sym_list[-1]

                sym_repr = final_sym_repr
                lin_repr = final_lin_repr

            else:
                raise NotImplementedError(f"cannot deal with `which={which}` for form replace.)")

            return Form(
                self.space,
                sym_repr,
                lin_repr,
                None,
            )

    def split(self, into):
        """reform self into a few forms."""
        raise NotImplementedError()
