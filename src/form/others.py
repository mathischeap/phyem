# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
@time: 11/26/2022 2:56 PM
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
from src.config import _global_operator_lin_repr_setting
from src.form.main import _global_forms, _global_form_variables


def _find_form(lin_repr, upon=None):
    """Find a form according to (pure) linguistic_representation.

    If we do not find such a form, we return None. Otherwise, we return the first found one.

    If upon is None, we seek the form whose either `sym_repr` or `lin_repr` is equal to `rp`.

    If upon is not None:
        If upon in _operators:
            We seek the form for which `upon(form)._lin_repr` is equal to `rp`.

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
            upon_name = upon.__name__
        else:
            raise NotImplementedError()

        if upon_name in _global_operator_lin_repr_setting:
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
        else:
            raise NotImplementedError()
    _global_form_variables['update_cache'] = True  # turn on cache! Very important!
    return the_one


def _list_forms(variable_range=None):
    """"""
    from src.config import RANK, MASTER_RANK
    if RANK != MASTER_RANK:
        return
    else:
        pass

    if variable_range is None:
        col_name_0 = 'form id'
    else:
        col_name_0 = 'variable name'

    if variable_range is None and len(_global_forms) >= 8:
        for form_id in _global_forms:
            form = _global_forms[form_id]
            print('--->', form_id, '|', form._sym_repr, '=', form._lin_repr)
    else:
        cell_text = list()
        for form_id in _global_forms:
            form = _global_forms[form_id]

            if variable_range is None:
                var_name = form_id
            else:
                var_name = list()
                for var in variable_range:
                    if variable_range[var] is form:
                        var_name.append(var)

                if len(var_name) == 0:  # a form is not involved in the variable_range.
                    continue
                elif len(var_name) == 1:
                    var_name = var_name[0]
                else:
                    var_name = ','.join(var_name)

            cell_text.append([r'\texttt{' + str(var_name) + '}',
                              rf"${form.space._sym_repr}$",
                              f"${form._sym_repr}$",
                              form._lin_repr,
                              form.is_root()])

        if len(cell_text) == 0:
            return
        else:
            pass

        fig, ax = plt.subplots(figsize=(16, (1 + len(cell_text))))
        fig.patch.set_visible(False)
        ax.axis('off')
        table = ax.table(cellText=cell_text, loc='center',
                         colLabels=[col_name_0, 'space', 'symbolic', 'linguistic', 'is_root()'],
                         colLoc='left', colColours='rgmcy',
                         cellLoc='left', colWidths=[0.15, 0.125, 0.125, 0.375, 0.075])
        table.scale(1, 8)
        table.set_fontsize(50)
        fig.tight_layout()
        from src.config import _setting, _pr_cache
        if _setting['pr_cache']:
            _pr_cache(fig, filename='formList')
        else:
            plt.show(block=_setting['block'])
        return fig
