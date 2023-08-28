# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from src.form.main import _global_root_forms_lin_dict
from src.config import _global_operator_lin_repr_setting, _non_root_lin_sep
from src.config import _global_lin_repr_setting, _root_form_ap_vec_setting
from src.config import _form_evaluate_at_repr_setting
from src.spaces.main import _VarSetting_boundary_dp_vector, _sep


class _BoundaryCondition(Frozen):
    """"""

    @classmethod
    def _check_raw_bc(cls, ls, raw_bc_form):
        """"""
        raise NotImplementedError(ls, raw_bc_form)


class _NaturalBoundaryCondition(_BoundaryCondition):
    """
    provide trace of star of a form on a boundary section.

    """
    def __init__(self, ls, raw_bc_form, i, j, provided_general_root_form, general_or_abstract):
        """"""
        self._ls = ls
        self._raw_bc_form = raw_bc_form
        assert isinstance(i, int) and 0 <= i < ls.b_shape[0], \
            f"i={i} is wrong, must be in [0, {ls.b_shape[0]-1}]"
        self._i = i
        assert isinstance(j, int) and j >= 0
        self._j = j
        if general_or_abstract is None:
            provided_root_form = provided_general_root_form
        else:
            the_abstract_root_form = None
            evaluate_lin = _form_evaluate_at_repr_setting['lin']
            for _pure_lin_repr in _global_root_forms_lin_dict:
                root_form = _global_root_forms_lin_dict[_pure_lin_repr]
                if evaluate_lin in root_form._pure_lin_repr:
                    if root_form._pure_lin_repr in general_or_abstract:
                        the_abstract_root_form = root_form
                else:
                    pass
            assert the_abstract_root_form is not None, f"must have found a abstract root-form."
            provided_root_form = the_abstract_root_form

        self._provided_root_form = provided_root_form  # i.e. the ``bf0``;
        # And the bc is ``<tr star bf0 | tr bf1>`` bf1 is the test root form.
        self._freeze()

    def __repr__(self):
        """"""
        super_repr = super().__repr__().split('object')[1]
        raw_bc_repr = self._raw_bc_form.__repr__()[1:].split(' at ')[0]
        return '<Natural bc: ' + raw_bc_repr + \
            f' through entry [{self.i}, {self.j}] of b' + super_repr

    @property
    def _abstract_array(self):
        """I am working in this abstract array, i.e. boldsymbol{b} shown in mp or ls."""
        return self._ls._b(self._i)[0][self._j]

    @property
    def _testing_form(self):
        """<~ | tr tf>

        This property gives the root abstract testing form, i.e. `tf` (not ``tr tf``).
        """
        start, end = _global_lin_repr_setting['array']
        pure_lin_repr = self._abstract_array._lin_repr[len(start):-len(end)]
        testing_form_part = pure_lin_repr.split(_sep)[2]
        template = _VarSetting_boundary_dp_vector[1].split(_sep)[2]

        testing_form = None
        for bf_lin in _global_root_forms_lin_dict:
            bf = _global_root_forms_lin_dict[bf_lin]

            if template.replace(r'{f1}', bf._pure_lin_repr) == testing_form_part:
                testing_form = bf
                break
            else:
                pass
        assert testing_form is not None, f"we must have found a testing form in {testing_form_part}"
        return testing_form

    def _pr_text(self):
        """"""
        text = r"$\mathrm{tr}\left(\star " + rf"{self._provided_root_form._sym_repr}\right)$ "
        text += r'in $\vec{b}[' + f"{self.i}][{self.j}] = " + self._abstract_array._sym_repr + "$ "
        text += r'provided as a \textbf{natural} boundary condition; '
        text += (r'$\left<\left.' + self._raw_bc_form._sym_repr + r'\right|\mathrm{tr}\ ' +
                 self._testing_form._sym_repr + r'\right>$')
        return text

    @property
    def i(self):
        """It is pointing the ith block of the b vector (in Ax = b).

        Returns
        -------

        """
        return self._i

    @property
    def j(self):
        """It is pointing the jth term of the ith block (see property ``i``)."""
        return self._j

    @classmethod
    def _check_raw_bc(cls, ls, raw_bc_form):
        """"""
        raw_bc_form_lin_repr = raw_bc_form._lin_repr

        evaluate_lin_repr = _form_evaluate_at_repr_setting['lin']

        if evaluate_lin_repr in raw_bc_form_lin_repr:
            # this raw_bc_form is not even a base one. It cannot match the
            return False

        else:
            pass

        trace_operator_lin_repr = _global_operator_lin_repr_setting['trace']
        Hodge_operator_lin_repr = _global_operator_lin_repr_setting['Hodge']

        template_lin_pattern = [
            trace_operator_lin_repr + _non_root_lin_sep[0] + Hodge_operator_lin_repr,
            _non_root_lin_sep[1]
        ]
        evaluate = _form_evaluate_at_repr_setting['lin']
        for root_lin_repr in _global_root_forms_lin_dict:

            full_template = root_lin_repr.join(template_lin_pattern)

            if full_template == raw_bc_form_lin_repr:

                # we have found the root-form (the general form) that works in this natural bc.

                root_form = _global_root_forms_lin_dict[root_lin_repr]

                # trace star of this root-form is the natural bc provided.

                b = ls._b

                target_lin_repr = _VarSetting_boundary_dp_vector[1]
                target_lin_repr = target_lin_repr.replace('{f0}', root_form._pure_lin_repr)
                target_header, target_lin_repr = target_lin_repr.split(_sep)[:2]
                target = target_header + _sep + target_lin_repr

                target = target[:-1]  # IMPORTANT: we remove the last `]` to make abstract form also works

                found_ij = list()
                general_or_abstract = list()
                for i, index in enumerate(b):
                    terms, signs = b(index)
                    for j, term in enumerate(terms):
                        if target in term._lin_repr:
                            found_ij.append((i, j, root_form))

                            general_or_abstract_in_b = term._lin_repr.split(_sep)[1]
                            if evaluate in general_or_abstract_in_b:
                                general_or_abstract.append(
                                    general_or_abstract_in_b
                                )
                            else:
                                general_or_abstract.append(
                                    None
                                )
                        else:
                            pass

                if len(found_ij) == 0:
                    return False
                elif len(found_ij) == 1:
                    i, j, base_root_form = found_ij[0]
                    return i, j, base_root_form, general_or_abstract[0]
                else:
                    raise Exception(f"found multiple entries for this natural bc, something is wrong!")

            else:
                pass

        return False


class _EssentialBoundaryCondition(_BoundaryCondition):
    """
    provide trace of a form on a boundary section.

    """
    def __init__(self, ls, raw_bc_form, i, provided_general_root_form, general_or_abstract):
        """"""
        self._ls = ls
        self._raw_bc_form = raw_bc_form
        assert isinstance(i, int) and 0 <= i < ls.b_shape[0], \
            f"i={i} is wrong, must be in [0, {ls.b_shape[0]-1}]"
        self._i = i
        if general_or_abstract is None:
            provided_root_form = provided_general_root_form
        else:
            the_abstract_root_form = None
            for _pure_lin_repr in _global_root_forms_lin_dict:
                _root_form = _global_root_forms_lin_dict[_pure_lin_repr]
                if _root_form._pure_lin_repr == general_or_abstract:
                    the_abstract_root_form = _root_form
                else:
                    pass
            assert the_abstract_root_form is not None, f"we must have found an abstract root form."
            provided_root_form = the_abstract_root_form

        self._provided_root_form = provided_root_form  # i.e. the ``bf0``.
        # And the bc is: ``trace bf0``.
        self._freeze()

    def __repr__(self):
        """"""
        super_repr = super().__repr__().split('object')[1]
        raw_bc_repr = self._raw_bc_form.__repr__()[1:].split(' at ')[0]
        return '<Essential bc: ' + raw_bc_repr + \
            f' through entry [{self.i}] of x' + super_repr

    def _pr_text(self):
        """"""
        unknown_vec_sym_repr = self._ls._x(self.i)[0][0]._sym_repr
        text = (r"$\mathrm{tr}\ " + rf"{self._provided_root_form._sym_repr}$ for " +
                r"$\vec{x}$" + f"[{self.i}] = ${unknown_vec_sym_repr}$ ")
        text += (r"provided as an \textbf{essential} boundary condition; $" +
                 self._raw_bc_form._sym_repr + '$')
        return text

    @property
    def i(self):
        """This boundary condition should be applied to the ith unknown."""
        return self._i

    @classmethod
    def _check_raw_bc(cls, ls, raw_bc_form):
        """

        Parameters
        ----------
        ls
        raw_bc_form
            The form given on a particular boundary section.

        Returns
        -------

        """
        trace_operator_lin_repr = _global_operator_lin_repr_setting['trace']
        trace_lin_repr_len = len(trace_operator_lin_repr)
        raw_bc_form_lin_repr = raw_bc_form._lin_repr

        if raw_bc_form_lin_repr[:trace_lin_repr_len] == trace_operator_lin_repr:

            the_rest_lin_repr = raw_bc_form_lin_repr[trace_lin_repr_len:]

            if the_rest_lin_repr in _global_root_forms_lin_dict:

                the_form = _global_root_forms_lin_dict[the_rest_lin_repr]

                if the_form._is_root and the_form._pAti_form['base_form'] is None:
                    # we have found the form, it is a root-form, and it has no base-form which means
                    # it is not a form evaluated at a particular time; thus it is an all-time form.

                    x = ls.x  # now we need to check whether the vector proxy of this form is in x of Ax=b.

                    found_indices = list()
                    general_or_abstract = list()

                    array_lin_repr_start, array_lin_repr_end = _global_lin_repr_setting['array']
                    vec_lin_repr = _root_form_ap_vec_setting['lin']
                    evaluate_lin_repr = _form_evaluate_at_repr_setting['lin']

                    looking_for = array_lin_repr_start + the_form._pure_lin_repr + \
                        vec_lin_repr + array_lin_repr_end

                    for i, index in enumerate(x):
                        x_vec, x_sign = x(index)
                        assert len(x_vec) == 1 and len(x_sign) == 1 and x_sign[0] == '+', \
                            f"x[{i}] = {x_sign[0]}{x_vec[0]} is wrong!"  # a safety check!

                        x_vec = x_vec[0]

                        if evaluate_lin_repr in x_vec._lin_repr:
                            # x_vec is ap of a form @ particular time-step.
                            evaluated = x_vec._lin_repr
                            evaluated = evaluated.split(vec_lin_repr)
                            evaluated[0] = evaluated[0].split(evaluate_lin_repr)[0]
                            x_vec_lin_repr = evaluated[0] + vec_lin_repr + evaluated[1]
                            _ = x_vec._pure_lin_repr.split(vec_lin_repr)[0]
                        else:  # x_vec is the ap of a general form.
                            x_vec_lin_repr = x_vec._lin_repr
                            _ = None

                        if x_vec_lin_repr == looking_for:
                            # this all-time essential boundary-condition will apply to this block-row!
                            found_indices.append(i)
                            general_or_abstract.append(_)
                        else:
                            pass

                    number_suit_rows = len(found_indices)
                    if number_suit_rows == 0:
                        raise Exception(
                            f"We found no unknown to accommodate this all-time essential bc: {raw_bc_form}."
                        )
                    elif number_suit_rows == 1:
                        pass
                    else:
                        raise Exception(f"found multiple rows, must be wrong!")

                    return found_indices[0], the_form, general_or_abstract[0]
                    # do not remove comma; return the *args for initialize an instance of this class.

                else:
                    pass

            else:
                pass

        else:
            pass

        return False


___all_boundary_type_classes___ = [
    _EssentialBoundaryCondition,
    _NaturalBoundaryCondition,
]


class MatrixProxyLinearSystemBoundaryConditions(Frozen):
    """"""

    def __init__(self, mp_ls, mp_bc):
        """"""
        self._ls = mp_ls._ls

        self._valid_bcs = {}

        for boundary_section_sym_repr in mp_bc:

            bcs = mp_bc[boundary_section_sym_repr]

            if len(bcs) == 0:  # section defined, but no valid boundary condition imposed on it.
                pass

            else:  # there are valid boundary conditions on this boundary section.

                self._valid_bcs[boundary_section_sym_repr] = list()

                for raw_bc_form in bcs:

                    bc_pattern = self._parse_bc_pattern(raw_bc_form)

                    self._valid_bcs[boundary_section_sym_repr].append(
                        bc_pattern
                    )

        self._freeze()

    def _parse_bc_pattern(self, raw_bc_form):
        """We study the raw bc item and retrieve the correct BoundaryCondition object here!

        If some boundary condition type is not recognized, raise Error here!

        ``raw_bc_form`` is a form, given on a particular boundary section.
        """
        found_pattern = None

        for btc in ___all_boundary_type_classes___:

            checks = btc._check_raw_bc(self._ls, raw_bc_form)

            if checks is False:
                pass

            else:

                found_pattern = btc(self._ls, raw_bc_form, *checks)

                break  # once found it, we break, because the boundary pattern as exclusive.

        if found_pattern is None:
            raise Exception(f'We cannot find a correct boundary condition '
                            f'pattern for raw bc form: {raw_bc_form}.')
        else:
            pass

        return found_pattern

    def __iter__(self):
        """go through all boundary section sym_repr that has valid BC on."""
        for boundary_sym_repr in self._valid_bcs:
            yield boundary_sym_repr

    def __getitem__(self, boundary_section_sym_repr):
        """Return the B.Cs on this boundary section."""
        assert boundary_section_sym_repr in self, \
            f"no valid BC is defined on {boundary_section_sym_repr}."
        return self._valid_bcs[boundary_section_sym_repr]

    def __len__(self):
        """How many valid boundary sections?"""
        return len(self._valid_bcs)

    def _bc_text(self):
        """"""
        text = '\n'
        i = 0
        for boundary_section in self:
            boundary_conditions = self[boundary_section]
            for bc in boundary_conditions:
                text += f"{i}) on $" + boundary_section + "$: " + bc._pr_text() + '\n'
                i += 1
        text = text[:-1]
        return text
