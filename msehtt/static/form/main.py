# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from typing import Dict
from src.form.main import Form
from src.config import RANK, MASTER_RANK, SIZE, COMM
import pickle
from msehtt.tools.matrix.static.local import MseHttStaticLocalMatrix
from msehtt.static.form.cf import MseHttStaticFormCF
from msehtt.static.form.addons.static import MseHttFormStaticCopy
from msehtt.static.form.addons.ic import MseHtt_From_InterpolateCopy
from msehtt.static.form.cochain.main import MseHttCochain
from msehtt.static.form.visualize.main import MseHttFormVisualize
from msehtt.static.form.bi.main import MseHttStaticForm_Boundary_Integrate
from msehtt.static.form.numeric.main import MseHtt_Form_Numeric
from msehtt.static.form.dofs.main import MseHtt_StaticForm_Dofs
from msehtt.static.form.print_.main import MseHtt_Static_Form_Print


class MseHttForm(Frozen):
    r""""""

    def __init__(self, abstract_root_form, SPECIAL=None):
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
        elif abstract_root_form is None and isinstance(SPECIAL, dict):   # to make an IMPLEMENTATION form.
            assert 'KEY' in SPECIAL, \
                f"Special form indicator must contain 'KEY' to tell which kind of special form we are saying."
            KEY = SPECIAL['KEY']
            if KEY == 'IMPLEMENTATION':             # IMPLEMENTATION form
                self._degree = SPECIAL['degree']
                self._abstract = None
                self._ftype = 'IMPLEMENTATION'
            else:
                raise NotImplementedError()

        else:
            raise Exception()

        self._name_ = None  # name of this form

        self._tgm = None  # the msehtt great mesh
        self._tpm = None  # the msehtt partial mesh
        self._space = None  # the msehtt space
        self._manifold = None   # the msehtt manifold

        self._cf = None
        self._cochain = None
        self._im = None
        self._bi = None
        self._numeric = None
        self._project = None
        self._dofs = None
        self._print_ = None

        self._freeze()

    @property
    def abstract(self):
        r"""Return the abstract form."""
        if self._abstract is None:
            raise Exception('No abstract form, DO NOT access it.')
        return self._abstract

    def __repr__(self):
        r"""repr"""
        if self._ftype == 'regular':
            ab_rf_repr = self.abstract.__repr__().split(' at ')[0][1:]
            return "<MseHtt " + ab_rf_repr + super().__repr__().split(" object")[1]
        elif self._ftype == 'IMPLEMENTATION':
            ab_rf_repr = 'IMPLEMENTATION-form'
            return "<MseHtt " + ab_rf_repr + super().__repr__().split(" object")[1]
        else:
            raise Exception()

    # ------------ IMPLEMENTATION FORM ---------------------------------------------------------------------

    def d(self):
        r"""Return AN IMPLEMENTATION FORM represent d(self)."""
        IMPLEMENTATION_NEXT_SPACE = self.space._IMPLEMENTATION_d_next_space_
        df = MseHttForm(
            None,
            SPECIAL={
                'KEY': 'IMPLEMENTATION',
                'degree': self.degree
            }
        )
        assert df._ftype == 'IMPLEMENTATION', f'must be!'
        df.space = IMPLEMENTATION_NEXT_SPACE
        df._tpm = self._tpm
        df._tgm = self._tgm
        df._manifold = self._manifold

        dcc = df.cochain

        for t in self.cochain:
            scc = self.cochain[t]
            dcc._set(t, scc.___coboundary_callable___)

        return df

    # ======================================================================================================

    # ---------- OPERATIONS --------------------------------------------------------------------------------

    def __sub__(self, other):
        r""""""
        if other.__class__ is self.__class__:
            if self._ftype == 'regular' and other._ftype == 'IMPLEMENTATION':
                return ___regular_sub_or_add_IMPLEMENTATION___(self, '-', other)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError(f"cannot do: {self.__class__} - {other.__class__}")

    def __add__(self, other):
        r""""""
        if other.__class__ is self.__class__:
            if self._ftype == 'regular' and other._ftype == 'IMPLEMENTATION':
                return ___regular_sub_or_add_IMPLEMENTATION___(self, '+', other)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError(f"cannot do: {self.__class__} + {other.__class__}")

    # ======================================================================================================

    @property
    def degree(self):
        r"""The degree of the form."""
        return self._degree

    def _is_base(self):
        r"""Am I a base root-form (not abstracted at a time)?"""
        return self._base is None

    @property
    def _base(self):
        r"""The base root-form I have.

        if `self._is_base()`, return None. Else return the base form.
        """
        return self._pAti_form['base_form']

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
        elif self._ftype == 'IMPLEMENTATION':
            assert space.abstract.__class__.__name__ == '_IMPLEMENTATION_AbstractSpace_'
            self._space = space
        else:
            raise NotImplementedError()

    @property
    def manifold(self):
        r"""Return the msehtt manifold."""
        return self._manifold

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
        else:
            self._base.cf = _cf

    @property
    def name(self):
        r""""""
        if self._is_base():
            if self._name_ is None:
                self._name_ = 'msehtt-form: ' + self.abstract._sym_repr + ' = ' + self.abstract._lin_repr
            return self._name_
        else:
            if self._name_ is None:
                base_name = self._base.name
                ati = self._pAti_form['ati']
                self._name_ = base_name + '@' + ati.__repr__()
            return self._name_

    def _read_cache_data(self, time_dict_data):
        r"""Only send data to master rank. Send None in others."""
        if RANK == MASTER_RANK:
            assert isinstance(time_dict_data, dict), f"I can only read dict."
            only_time_data_dict = {}
            original_global_element_signature_dict = None
            all_times = []
            for key in time_dict_data:
                if key == 'global_element_signature_dict':
                    original_global_element_signature_dict = time_dict_data[key]
                else:
                    all_times.append(key)
                    only_time_data_dict[key] = time_dict_data[key]
            assert len(only_time_data_dict) > 0, f"I receive no time data"
            assert original_global_element_signature_dict is not None, \
                f"I found no original_global_element_signature_dict"
        else:
            all_times = None
            assert time_dict_data is None, f"to other ranks, pls send None."

        all_times = COMM.bcast(all_times, root=MASTER_RANK)
        Element_signature_dict = COMM.gather(self.element_signature_dict, root=MASTER_RANK)

        if RANK == MASTER_RANK:
            LOCAL_COCHAIN = dict()
            # noinspection PyUnboundLocalVariable
            for t in only_time_data_dict:
                original_cochain_t = only_time_data_dict[t]
                LOCAL_COCHAIN_t = []
                for rank in range(SIZE):
                    element_signature_dict = Element_signature_dict[rank]
                    local_cochain_t = {}
                    for element_signature in element_signature_dict:
                        current_e = element_signature_dict[element_signature]
                        # noinspection PyUnboundLocalVariable
                        original_e = original_global_element_signature_dict[element_signature]
                        local_cochain_t[current_e] = original_cochain_t[original_e]
                    LOCAL_COCHAIN_t.append(local_cochain_t)
                LOCAL_COCHAIN[t] = LOCAL_COCHAIN_t
        else:
            pass

        for t in all_times:
            if RANK == MASTER_RANK:
                # noinspection PyUnboundLocalVariable
                local_cochain = COMM.scatter(LOCAL_COCHAIN[t], root=MASTER_RANK)
            else:
                # noinspection PyTypeChecker
                local_cochain = COMM.scatter(None, root=MASTER_RANK)

            self.cochain._set(t, local_cochain)

    def _make_cache_data(self, t=None):
        r"""Return all data in master rank, return None in others."""
        if t is None:
            times = [self.cochain.newest, ]  # do it for the newest time only

        elif (
            isinstance(t, tuple) and len(t) == 2 and
            isinstance(t[0], str) and t[0] == 'nearest' and isinstance(t[1], int)
        ):
            # we receive an indicator saying we need to make cache data for several nearest data
            # like t = ('nearest', 2)
            if t[1] == 1:
                times = [self.cochain.newest, ]
            else:
                raise NotImplementedError()

        else:
            raise NotImplementedError(f"t={t} cannot be parsed.")

        cache_data = {}
        for t in times:
            t = self.cochain._parse_t(t)  # round off the truncation error to make it clear.
            data = self._collect_cache_data_at_t(t)
            cache_data[t] = data

        cache_data['global_element_signature_dict'] = self.global_element_signature_dict

        if RANK == MASTER_RANK:
            return cache_data
        else:
            return None

    def _collect_cache_data_at_t(self, t):
        r"""Return data in master rank, return None in others."""
        if self._is_base():
            pass
        else:
            return self._base._collect_cache_data_at_t(t)

        sf = self[t]
        total_cochain = sf.cochain._merge_to(root=MASTER_RANK)
        return total_cochain

    @property
    def global_element_signature_dict(self):
        r"""Only return it in Master Rank, in other ranks, return None"""
        Element_signature_dict = COMM.gather(self.element_signature_dict, root=MASTER_RANK)

        if RANK == MASTER_RANK:
            Element_Signature_Dict = {}
            for _ in Element_signature_dict:
                Element_Signature_Dict.update(_)

            return Element_Signature_Dict
        else:
            return None

    @property
    def element_signature_dict(self):
        r"""Return local element signature dict. Keys are element signatures, values are element indices."""
        element_signature_dict = {}
        for e in self.tpm.composition:
            element = self.tpm.composition[e]
            element_signature_dict[element.signature] = e
        return element_signature_dict

    def saveto(self, filename, what=None):
        """save me to a file.

        Basically, we only save the co-chains of all available times.
        """
        if self._is_base():
            pass
        else:
            self._base.saveto(filename, what=what)
            return

        element_signature_dict = {}
        for e in self.tpm.composition:
            element = self.tpm.composition[e]
            element_signature_dict[element.signature] = e

        if what is None:  # save all existing cochain
            time_range = self.cochain._tcd.keys()
        elif isinstance(what, int) and what > 0:  # save the newest `what` cochain
            time_range = list(self.cochain._tcd.keys())
            time_range.sort()
            if len(time_range) <= what:
                pass
            else:
                time_range = time_range[-what:]
        else:
            raise NotImplementedError(
                f"what={what} is cannot be saved. When `what` is a positive integer,"
                f"we save the newest this amount of cochain. When `what` is None, we"
                f"save all cochain.")

        cochain = {}
        for t in time_range:
            sf = self[t]
            total_cochain = sf.cochain._merge_to(root=MASTER_RANK)
            cochain[t] = total_cochain

        element_signature_dict = COMM.gather(element_signature_dict, root=MASTER_RANK)

        if RANK == MASTER_RANK:
            Element_Signature_Dict = {}
            for _ in element_signature_dict:
                Element_Signature_Dict.update(_)
            del element_signature_dict
            form_para_dict = {
                'key': 'msehtt-static-form',
                'name': self.name,
                'cochain': cochain,
                'element signature dict': Element_Signature_Dict
            }
            with open(filename, 'wb') as output:
                # noinspection PyTypeChecker
                pickle.dump(form_para_dict, output, pickle.HIGHEST_PROTOCOL)
            output.close()
        else:
            pass

    def read(self, filename):
        """Read to my cochain from a file whose key is equal to 'msehtt-static-form'.

        Since I am only reading to my cochain, it returns no new object, it cannot be called
        from `ph.read`. Please call me from a particular form.

        """

        element_signature_dict = {}
        for e in self.tpm.composition:
            element = self.tpm.composition[e]
            element_signature_dict[e] = element.signature

        for rank in range(SIZE):
            if RANK == rank:
                with open(filename, 'rb') as inputs:
                    obj = pickle.load(inputs)
                inputs.close()
                assert obj['key'] == 'msehtt-static-form'
                local_cochain = {}
                cochain = obj['cochain']
                Element_Signature_Dict = obj['element signature dict']
                for t in cochain:
                    total_t_cochain = cochain[t]
                    local_t_cochain = {}
                    for e in self.cochain.gathering_matrix:
                        element_signature = element_signature_dict[e]
                        assert element_signature in Element_Signature_Dict, \
                            f"Miss cochain for element indexed {e}: {self.tpm.composition[e]}"
                        index_in_file = Element_Signature_Dict[element_signature]
                        local_t_cochain[e] = total_t_cochain[index_in_file]
                    local_cochain[t] = local_t_cochain
                del obj
            else:
                pass
            COMM.barrier()  # make sure the file is read one rank by one rank.

        # noinspection PyUnboundLocalVariable
        for t in local_cochain:
            self.cochain._set(t, local_cochain[t])

    def __getitem__(self, t):
        """"""
        if t is None:
            t = self.cochain.newest  # newest time
        elif isinstance(t, str):
            # when use str, we are looking for the form at a time step.
            from src.time_sequence import _global_abstract_time_sequence
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

        t = self.cochain._parse_t(t)  # round off the truncation error to make it clear.

        if isinstance(t, (int, float)):
            if self._is_base():
                return MseHttFormStaticCopy(self, t)
            else:
                return MseHttFormStaticCopy(self._base, t)
        else:
            raise Exception(f"cannot accept t={t}.")

    def __call__(self, t, extrapolate=False):
        """"""
        if isinstance(t, str):
            # when use str, we are looking for the form at a time step.
            from src.time_sequence import _global_abstract_time_sequence
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

        t = self.cochain._parse_t(t)

        if isinstance(t, (int, float)):
            if self._is_base():
                return MseHtt_From_InterpolateCopy(self, t, extrapolate=extrapolate)
            else:
                return MseHtt_From_InterpolateCopy(self._base, t, extrapolate=extrapolate)
        else:
            raise Exception(f"cannot accept t={t}.")

    @property
    def cochain(self):
        """The cochain class."""
        if self._cochain is None:
            self._cochain = MseHttCochain(self)
        return self._cochain

    def reduce(self, cf_at_t):
        """"""
        return self._space.reduce(cf_at_t, self.degree)

    def reconstruct(self, cochain, *meshgrid, ravel=False, element_range=None):
        """"""
        return self._space.reconstruct(self.degree, cochain, *meshgrid, ravel=ravel, element_range=element_range)

    def error(self, cf, cochain, error_type='L2'):
        """"""
        return self._space.error(cf, cochain, self.degree, error_type=error_type)

    def visualize(self, t):
        if self._is_base():
            return MseHttFormVisualize(self, t)
        else:
            return self._base.visualize(t)

    @property
    def incidence_matrix(self):
        if self._im is None:
            gm0 = self.space.gathering_matrix._next(self.degree)
            gm1 = self.cochain.gathering_matrix
            E, cache_key_dict = self.space.incidence_matrix(self.degree)
            self._im = MseHttStaticLocalMatrix(
                E,
                gm0,
                gm1,
                cache_key=cache_key_dict
            )
        return self._im

    def norm(self, cochain, norm_type='L2', component_wise=False):
        """"""
        return self.space.norm(self.degree, cochain, norm_type=norm_type, component_wise=component_wise)

    def inner_product(
            self, self_cochain, other_form, other_degree, other_cochain, inner_type='L2'):
        r"""

        Parameters
        ----------
        self_cochain
        other_form
        other_degree
        other_cochain
        inner_type

        Returns
        -------

        """
        assert other_form.__class__ is self.__class__, f"other-form must be a form."
        if self._tpm is other_form._tpm:   # on the same grid.
            return self.space.inner_product(
                self.degree, self_cochain,
                other_form.space, other_degree, other_cochain,
                inner_type=inner_type,
            )
        else:
            raise NotImplementedError(f"cross adaptive grids? to be implemented")

    @property
    def bi(self):
        """boundary integrate."""
        if self._bi is None:
            self._bi = MseHttStaticForm_Boundary_Integrate(self)
        return self._bi

    def reconstruction_matrix(self, *meshgrid):
        """Return the reconstruction matrix for all rank elements."""
        return self.space.reconstruction_matrix(self.degree, *meshgrid)

    @property
    def numeric(self):
        """"""
        if self._is_base():
            if self._numeric is None:
                self._numeric = MseHtt_Form_Numeric(self)
            return self._numeric
        else:
            return self._base.numeric

    def norm_residual(self, from_time=None, to_time=None, norm_type='L2'):
        """By default, use L2-norm."""
        if to_time is None or from_time is None:
            all_cochain_times = list(self.cochain._tcd.keys())
            all_cochain_times.sort()

            if to_time is None:
                if len(all_cochain_times) == 0:
                    to_time = None
                else:
                    to_time = all_cochain_times[-1]
            else:
                pass

            if from_time is None:
                if len(all_cochain_times) == 0:
                    from_time = None
                elif len(all_cochain_times) == 1:
                    from_time = all_cochain_times[-1]
                else:
                    from_time = all_cochain_times[-2]
            else:
                pass

        else:
            pass

        if from_time is None:
            assert to_time is None, f"cochain must be of no cochain."
            return np.nan  # we return np.nan.
        else:
            assert isinstance(from_time, (int, float)), f"from_time = {from_time} is wrong."
            assert isinstance(to_time, (int, float)), f"to_time = {to_time} is wrong."
            from_time = self.cochain._parse_t(from_time)
            to_time = self.cochain._parse_t(to_time)

            if from_time == to_time:
                return np.nan  # return nan since it is not a residual

            else:
                from_cochain = self.cochain[from_time]
                to_cochain = self.cochain[to_time]
                diff_cochain = to_cochain - from_cochain

                # for i in diff_cochain:
                #     print(i, to_cochain[i])

                norm = self.space.norm(self.degree, diff_cochain, norm_type=norm_type)
                return norm

    @property
    def dofs(self):
        r"""A property that wraps all geometric properties of degrees of freedom."""
        if self._is_base():
            if self._dofs is None:
                self._dofs = MseHtt_StaticForm_Dofs(self)
            return self._dofs
        else:
            return self._base.dofs

    @property
    def print(self):
        r"""To print like properties, attributes or else of this form. It is usually for checking/debugging purpose."""
        if self._print_ is None:
            self._print_ = MseHtt_Static_Form_Print(self)
        return self._print_


# ========================= other functions ===============================================================


def ___regular_sub_or_add_IMPLEMENTATION___(rgf, sub_or_add, IMf):
    r"""regular form - IMPLEMENTATION form, return an IMPLEMENTATION form."""
    assert rgf._ftype == 'regular', f'must be!'
    assert IMf._ftype == 'IMPLEMENTATION', f'must be!'

    if sub_or_add == '-':
        sub = True
    elif sub_or_add == '+':
        sub = False
    else:
        raise Exception()

    s0 = rgf.space
    s1 = IMf.space
    assert s0._imn_ == s1._imn_, f"The two forms have different spaces. {s0._imn_} != {s1._imn_}."
    assert rgf.degree == IMf.degree, f"degree must be "
    indicator = s0._imn_[0]
    if indicator == 'Lambda':
        k0, k1 = s0.abstract.k, s1.abstract.k
        o0, o1 = s0.abstract.orientation, s1.abstract.orientation
        assert k0 == k1 and o0 == o1, \
            f"the k and orientation are different; k0={k0}, k1={k1}, o0={o0}, o1={o1}"
        SUBf = MseHttForm(
            None,
            SPECIAL={
                'KEY': 'IMPLEMENTATION',
                'degree': rgf.degree
            }
        )
        assert SUBf._ftype == 'IMPLEMENTATION', f'must be!'
        SUBf.space = IMf.space
        SUBf._tpm = rgf._tpm
        SUBf._tgm = rgf._tgm
        SUBf._manifold = rgf._manifold

        cc0 = rgf.cochain
        cc1 = IMf.cochain
        for t in cc0:
            if t in cc1:
                ccI0 = cc0[t]
                ccI1 = cc1[t]
                if sub:
                    _ = ccI0 - ccI1
                else:
                    _ = ccI0 + ccI1
                SUBf.cochain._set(t, _)

        return SUBf

    else:
        raise NotImplementedError()
