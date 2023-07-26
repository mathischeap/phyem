# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 2:00 PM on 7/5/2023
"""
from tools.frozen import Frozen
from src.form.main import _global_root_forms_lin_dict
from src.config import _global_operator_lin_repr_setting, _non_root_lin_sep
from src.config import _global_lin_repr_setting, _root_form_ap_vec_setting
from src.config import _form_evaluate_at_repr_setting
from src.spaces.main import _default_boundary_dp_vector_reprs, _sep


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
    def __init__(self, ls, raw_bc_form, i, j, provided_root_form):
        """"""
        self._ls = ls
        self._raw_bc_form = raw_bc_form
        assert isinstance(i, int) and 0 <= i < ls.b_shape[0], \
            f"i={i} is wrong, must be in [0, {ls.b_shape[0]-1}]"
        self._i = i
        assert isinstance(j, int) and j >= 0
        self._j = j
        self._provided_root_form = provided_root_form  # i.e. the ``bf0``;
        # And the bc is ``<tr star bf0 | tr bf1>`` bf1 is the test root form.
        self._freeze()

    def __repr__(self):
        """"""
        super_repr = super().__repr__().split('object')[1]
        raw_bc_repr = self._raw_bc_form.__repr__()[1:].split(' at ')[0]
        return '<Natural bc: ' + raw_bc_repr + \
            f' through entry [{self.i}, {self.j}] of b' + super_repr

    def _pr_text(self):
        """"""
        the_involving_term = self._ls._b(self._i)[0][self._j]
        text = r"$\mathrm{tr}\left(\star" + rf"{self._provided_root_form._sym_repr}\right)$ "
        text += 'in $' + the_involving_term._sym_repr + r'$ provided as a \textbf{natural} boundary condition.'
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

        for root_lin_repr in _global_root_forms_lin_dict:

            full_template = root_lin_repr.join(template_lin_pattern)

            if full_template == raw_bc_form_lin_repr:

                # we have found the root-form that works in this natural bc.

                root_form = _global_root_forms_lin_dict[root_lin_repr]

                # trace star of this root-form is the natural bc provided.

                b = ls._b

                target_lin_repr = _default_boundary_dp_vector_reprs[1]
                target_lin_repr = target_lin_repr.replace('{f0}', root_form._pure_lin_repr)
                target_header, target_lin_repr = target_lin_repr.split(_sep)[:2]
                target = target_header + _sep + target_lin_repr
                found_ij = list()
                for i, index in enumerate(b):
                    terms, signs = b(index)
                    for j, term in enumerate(terms):
                        if target in term._lin_repr:
                            found_ij.append((i, j, root_form))

                if len(found_ij) == 0:
                    return False
                elif len(found_ij) == 1:
                    return found_ij[0]
                else:
                    raise Exception(f"found multiple entries for this natural bc, something is wrong!")

            else:
                pass

        return False


class _EssentialBoundaryCondition(_BoundaryCondition):
    """
    provide trace of a form on a boundary section.

    """
    def __init__(self, ls, raw_bc_form, i, provided_root_form):
        """"""
        self._ls = ls
        self._raw_bc_form = raw_bc_form
        assert isinstance(i, int) and 0 <= i < ls.b_shape[0], \
            f"i={i} is wrong, must be in [0, {ls.b_shape[0]-1}]"
        self._i = i
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
        text = r"$\mathrm{tr}\ " + rf"{self._provided_root_form._sym_repr}$ "
        text += r"provided as an \textbf{essential} boundary condition."
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

                        if evaluate_lin_repr in x_vec._lin_repr:  # x_vec is ap of a form @ particular time-step.
                            raise NotImplementedError()
                        else:  # x_vec is the ap of a general form.
                            x_vec_lin_repr = x_vec._lin_repr

                        if x_vec_lin_repr == looking_for:
                            # this all-time essential boundary-condition will apply to this block-row!
                            found_indices.append(i)
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

                    return found_indices[0], the_form
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
