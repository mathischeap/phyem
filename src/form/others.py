# -*- coding: utf-8 -*-
r"""
"""
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "DejaVu Sans",
    "text.latex.preamble": r"\usepackage{amsmath}"
})
matplotlib.use('TkAgg')

from types import FunctionType

from phyem.src.config import _global_operator_lin_repr_setting
from phyem.src.config import _non_root_lin_sep
from phyem.src.form.main import _global_forms, _global_form_variables


def _find_root_forms_through_lin_repr(lin_repr):
    """"""
    forms = list()
    for form_in in _global_forms:
        form = _global_forms[form_in]
        if form._is_root and form._lin_repr in lin_repr:
            forms.append(form)
    return forms


def _find_forms_through_lin_repr(lin_repr):
    """"""
    forms = list()
    for form_in in _global_forms:
        form = _global_forms[form_in]
        if form._lin_repr in lin_repr:
            forms.append(form)
    return forms


def _find_form(lin_repr, upon=None):
    """Find a form according to (pure) linguistic_representation.

    If we do not find such a form, we return None. Otherwise, we return the first found one.

    If upon is None, we seek the form whose either `sym_repr` or `lin_repr` is equal to `lin_repr`.

    If upon is not None:
        If upon in _operators:
            We seek the form for which `upon(form)._lin_repr` is equal to `lin_repr`.

    """
    # during this process, we do not cache the intermediate forms.
    _global_form_variables['update_cache'] = False

    if upon is None:
        the_one = None
        for form_id in _global_forms:
            form = _global_forms[form_id]
            if form._lin_repr == lin_repr or form._pure_lin_repr == lin_repr:
                the_one = form
                break
            else:
                pass

    else:
        if isinstance(upon, FunctionType):
            upon_type = "func-call"
            upon_name = upon.__name__
        elif isinstance(upon, str):
            upon_type = "operator-name"
            upon_name = upon
        else:
            raise NotImplementedError()

        if upon_type == "func-call" and upon_name in _global_operator_lin_repr_setting:
            the_one = None
            for form_id in _global_forms:
                form = _global_forms[form_id]

                try:
                    operator_of_f = upon(form)

                except:
                    continue

                else:
                    if operator_of_f._lin_repr == lin_repr or operator_of_f._pure_lin_repr == lin_repr:
                        the_one = form
                        break
                    else:
                        pass

        elif upon_type == "operator-name" and upon_name in _global_operator_lin_repr_setting:
            the_one = None

            op_lin_repr = _global_operator_lin_repr_setting[upon_name]

            if isinstance(op_lin_repr, str):  # this operator lin repr is a single str.

                for form_id in _global_forms:
                    form = _global_forms[form_id]
                    lr = form._lin_repr

                    operator_of_non_root_f = op_lin_repr + _non_root_lin_sep[0] + lr + _non_root_lin_sep[1]
                    operator_of_f = op_lin_repr + lr

                    if operator_of_f == lin_repr or operator_of_non_root_f == lin_repr:
                        the_one = form
                        break
                    else:
                        pass

            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    if the_one is None:
        nr0, nr1 = _non_root_lin_sep
        if upon is None:
            for form_id in _global_forms:
                form = _global_forms[form_id]

                if nr0 + form._lin_repr + nr1 == lin_repr:
                    the_one = form
                    break
                    the_one = form
                    break
                elif form._lin_repr == nr0 + lin_repr:
                    the_one = form
                    break
                elif form._lin_repr == lin_repr + nr1:
                    the_one = form
                    break
                else:
                    pass

        else:
            pass
    else:
        pass

    # if the_one is None:
    #     for form_id in _global_forms:
    #         form = _global_forms[form_id]
    #         if lin_repr in form._lin_repr:
    #             print(lin_repr, form._lin_repr)

    _global_form_variables['update_cache'] = True  # turn on cache! Very important!
    return the_one


def _list_forms():
    """"""
    from phyem.src.config import RANK, MASTER_RANK
    if RANK != MASTER_RANK:
        return None
    else:
        pass

    cell_text = list()
    for form_id in _global_forms:
        form = _global_forms[form_id]
        # noinspection PyTypeChecker
        cell_text.append(
            r'$\quad$'.join(
                [
                    r'\texttt{' + str(form_id) + '}',
                    rf"${form.space._sym_repr}$",
                    f"${form._sym_repr}$",
                    form._lin_repr,
                    'root' if form.is_root() else 'not-root'
                ]
            )
        )

    from phyem.src.wf.term.main import _global_wf_terms
    for term_id in _global_wf_terms:
        wft = _global_wf_terms[term_id]
        # noinspection PyTypeChecker
        cell_text.append(
            r'$\quad$'.join(
                [
                    r'\texttt{' + str(term_id) + '}',
                    rf"${wft._sym_repr}$",
                    wft._lin_repr
                ]
            )
        )

    if len(cell_text) == 0:
        return None
    else:
        pass

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_visible(True)
    ax.axis('off')
    cell_text = '\n'.join(cell_text)
    plt.text(0, 1, cell_text, va='top', ha='left', fontsize=20)
    from phyem.src.config import _setting, _pr_cache
    if _setting['pr_cache']:
        _pr_cache(fig, filename='formList')
    else:
        fig.tight_layout()
        plt.show(block=_setting['block'])
    return fig
