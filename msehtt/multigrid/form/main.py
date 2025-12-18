r""""""
from typing import Dict

from phyem.tools.frozen import Frozen
from phyem.src.form.main import Form
from phyem.msehtt.static.form.main import MseHttForm

from phyem.msehtt.static.form.cf import MseHttStaticFormCF
from phyem.msehtt.multigrid.form.addons.static import MseHtt_MultiGrid_FormStaticCopy


class MseHtt_MultiGrid_Form(Frozen):
    r""""""

    def __init__(self, abstract_root_form):
        r""""""
        self._pAti_form: Dict = {
            'base_form': None,  # the base form
            'ats': None,  # abstract time sequence
            'ati': None,  # abstract time instant
        }
        self._ats_particular_forms = dict()   # the abstract forms based on this form.
        if abstract_root_form.__class__ is Form:
            self._degree = abstract_root_form._degree
            self._abstract = abstract_root_form
            self._ftype = 'regular'
        else:
            raise Exception()

        self._name_ = None  # name of this form

        self._tgm = None  # the msehtt great mesh
        self._tpm = None  # the msehtt partial mesh
        self._space = None  # the msehtt space
        self._manifold = None   # the msehtt manifold

        self._levels = dict()

        self._cf = None

        self._freeze()

    @property
    def abstract(self):
        r"""Return the abstract form."""
        if self._abstract is None:
            raise Exception('No abstract form, DO NOT access it.')
        return self._abstract

    @property
    def degree(self):
        r"""The degree of the form."""
        return self._degree

    def __repr__(self):
        r"""repr"""
        if self._ftype == 'regular':
            ab_rf_repr = self.abstract.__repr__().split(' at ')[0][1:]
            return "<MseHtt-MultiGrid " + ab_rf_repr + super().__repr__().split(" object")[1]
        else:
            raise Exception()

    @property
    def _base(self):
        r"""The base root-form I have.

        if `self._is_base()`, return None. Else return the base form.
        """
        return self._pAti_form['base_form']

    def _is_base(self):
        r"""Am I a base root-form (not abstracted at a time)?"""
        return self._base is None

    @property
    def name(self):
        r""""""
        if self._is_base():
            if self._name_ is None:
                self._name_ = 'msehtt-multi-grid-form: ' + self.abstract._sym_repr + ' = ' + self.abstract._lin_repr
            return self._name_
        else:
            if self._name_ is None:
                base_name = self._base.name
                ati = self._pAti_form['ati']
                self._name_ = base_name + '@' + ati.__repr__()
            return self._name_

    @property
    def tgm(self):
        r"""Return the msehtt great mesh."""
        return self._tgm

    @property
    def tpm(self):
        r"""Return the msehtt partial mesh."""
        return self._tpm

    @property
    def space(self):
        r"""Return the msehtt space."""
        return self._space

    @space.setter
    def space(self, space):
        r""""""
        if self._ftype == 'regular':
            self._space = space
        else:
            raise NotImplementedError()

    @property
    def manifold(self):
        r"""Return the msehtt manifold."""
        return self._manifold

    def ___make_multigrid_level_degree_modifier___(self, lvl):
        r"""When the degree on levels are different (could be different p or dtype), we will use this
        method to find the correct degree modifier which will be sent to forms on particular levels.

        Then, when we parse the degree of a particular form, we will study the degree modifier (appended
        at the end of the degree) to finally get a correct p and dtype.
        """
        degree = self.degree
        if isinstance(degree, str) and degree[:2] == 'MG':
            tgm = self._tgm
            refining_method =tgm._configuration['method']
            if refining_method == 'uniform':
                parameters = tgm._configuration['parameters']
                rff = parameters['rff']
                max_levels = parameters['max-levels']
                return f"lvl:{lvl}=method:uniform=rff:{rff}=max_levels:{max_levels}"
            else:
                raise NotImplementedError()
        else:
            return None

    def get_level(self, lvl=None):
        r"""Return the msehtt form on the lvl-th level."""
        if lvl is None:
            lvl = self.tgm.max_level
        else:
            pass
        if lvl in self._levels:
            return self._levels[lvl]
        else:
            if self._is_base():

                mg_lvl_degree_modifier = self.___make_multigrid_level_degree_modifier___(lvl)

                lvl_base_form = MseHttForm(self.abstract)
                lvl_base_form.___modify_mg_level_degree___(mg_lvl_degree_modifier)
                lvl_base_form._tgm = self.tgm.get_level(lvl)
                lvl_base_form._tpm = self.tpm.get_level(lvl)
                lvl_base_form._manifold = self.manifold
                lvl_base_form._space = self.space.get_level(lvl)
                self._levels[lvl] = lvl_base_form
                for lin_repr in self._ats_particular_forms:
                    multi_great_particular_form = self._ats_particular_forms[lin_repr]
                    assert multi_great_particular_form.__class__ is self.__class__
                    assert not multi_great_particular_form._is_base()
                    ats = multi_great_particular_form._pAti_form['ats']
                    ati = multi_great_particular_form._pAti_form['ati']
                    assert ats.__class__.__name__ == "AbstractTimeSequence"
                    assert ati.__class__.__name__ == "AbstractTimeInstant"
                    prf = MseHttForm(multi_great_particular_form.abstract)
                    prf.___modify_mg_level_degree___(mg_lvl_degree_modifier)

                    prf._pAti_form['base_form'] = lvl_base_form
                    prf._pAti_form['ats'] = ats
                    prf._pAti_form['ati'] = ati

                    prf._tgm = lvl_base_form._tgm
                    prf._tpm = lvl_base_form._tpm
                    prf._manifold = lvl_base_form._manifold
                    prf._space = lvl_base_form._space

                    lvl_base_form._ats_particular_forms[lin_repr] = prf

            else:
                lvl_base_form = self._base.get_level(lvl)
                self._levels[lvl] = lvl_base_form._ats_particular_forms[self.abstract._lin_repr]

            return self._levels[lvl]

    @property
    def cf(self):
        r"""Continuous form."""
        if self._is_base():
            if self._cf is None:
                self._cf = MseHttStaticFormCF(self)
            return self._cf
        else:
            return self._base.cf

    @cf.setter
    def cf(self, _cf):
        r""""""
        if self._is_base():
            self.cf.field = _cf

            for lvl in self.tgm.level_range:
                self.get_level(lvl).cf.field = _cf

        else:
            self._base.cf = _cf

    def __getitem__(self, t):
        r""""""
        if t is None:
            t = self.get_level().cochain.newest  # newest time
        elif isinstance(t, str):
            # when use str, we are looking for the form at a time step.
            from phyem.src.time_sequence import _global_abstract_time_sequence
            if len(_global_abstract_time_sequence) == 1:
                ts_indicator = list(_global_abstract_time_sequence.keys())[0]
                ts = _global_abstract_time_sequence[ts_indicator]
                abstract_time_instant = ts[t]
                t = abstract_time_instant()()
            else:
                raise Exception(f"multiple time sequences exist. "
                                f"I don't know which one you are referring to")
        else:
            pass

        t = self.get_level().cochain._parse_t(t)  # round off the truncation error to make it clear.

        if isinstance(t, (int, float)):
            if self._is_base():
                return MseHtt_MultiGrid_FormStaticCopy(self, t)
            else:
                return MseHtt_MultiGrid_FormStaticCopy(self._base, t)
        else:
            raise Exception(f"cannot accept t={t}.")
