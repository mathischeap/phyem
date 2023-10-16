# -*- coding: utf-8 -*-
r"""
"""
from typing import Dict
from tools.frozen import Frozen
from _MPI.generic.py._2d_unstruct.form.main import MPI_Py_2D_Unstructured_Form
from _MPI.msehy.py._2d.space.main import MPI_MseHy_Py2_Space

from _MPI.msehy.py._2d.form.numeric.main import MPI_MseHy_Py2_Form_Numeric


class MPI_MseHy_Py2_Form(Frozen):
    """"""

    def __init__(self, abstract_root_form):
        """"""
        self._abstract = abstract_root_form
        abstract_space = abstract_root_form.space
        self._space = abstract_space._objective
        assert self.space.__class__ is MPI_MseHy_Py2_Space, f"space {self.space} must be a {MPI_MseHy_Py2_Space}."
        degree = self.abstract._degree
        assert degree is not None, f"{abstract_root_form} has no degree."
        self._degree = degree
        self._pAti_form: Dict = {
            'base_form': None,   # the base form
            'ats': None,   # abstract time sequence
            'ati': None,   # abstract time instant
        }
        self._ats_particular_forms = dict()   # the abstract forms based on this form.

        # --- checks --------------------------------------------------------------
        assert self.mesh.generation == 0, f'When initialize msehy-py2 form, the msehy-py mesh must be on G0'
        self._generation = None
        self._previous = None
        self._generic = None
        self._do_initialize = True

        # -------- properties -----------------------------------------------------
        self._cf_cache = None
        self._numeric = MPI_MseHy_Py2_Form_Numeric(self)

        self._freeze()

    def __repr__(self):
        """repr"""
        space_repr = self.space.__repr__().split(' at ')[0]
        return f"<Form {self._abstract._sym_repr}" + f' in {space_repr}>'

    def is_dual_representation(self):
        return self.abstract.is_dual_representation()

    @property
    def abstract(self):
        """the abstract object this root-form is for."""
        return self._abstract

    @property
    def space(self):
        """refer to the generic space."""
        return self._space

    @property
    def name(self):
        """name of this form is the pure linguistic representation."""
        return self._abstract._pure_lin_repr

    @property
    def degree(self):
        """The degree of my space."""
        return self._degree

    @property
    def mesh(self):
        """refer to the generic mesh"""
        return self.space.mesh

    def _is_base(self):
        """Am I a base root-form (not abstracted at a time.)"""
        return self._base is None

    @property
    def _base(self):
        """The base root-form I have."""
        return self._pAti_form['base_form']

    # --- properties -------------------------------------------------------------------------
    @property
    def cf(self):
        return self.generic._cf

    @cf.setter
    def cf(self, _cf):
        self._cf_cache = _cf
        self.generic.cf = _cf

    def __getitem__(self, time):
        """"""
        return self.generic[time]

    # ----------- generic ---------------------------------------------------------------------
    @property
    def generation(self):
        if self._do_initialize:
            self._initialize()
        return self._generation

    @property
    def generic(self):
        if self._do_initialize:
            self._initialize()
        return self._generic

    @property
    def previous(self):
        if self._do_initialize:
            self._initialize()
        return self._previous

    def _initialize(self):
        """update generation, generic, previous according to the newest generation of the mesh."""
        if self._do_initialize:
            self.space._initialize()
            self._generic = MPI_Py_2D_Unstructured_Form(
                self.space.generic, self.degree,
                dual_representation=self.is_dual_representation(),
                **self._pAti_form
            )
            self._generic._name = self.name

            self._generation = self.space.generation
            if self.space.previous is None:
                assert self._generation == 0, 'must be!'
            else:
                self._previous = MPI_Py_2D_Unstructured_Form(
                    self.space.previous, self.degree,
                    dual_representation=self.is_dual_representation(),
                    **self._pAti_form
                )
                self._previous._name = self.name

            self._do_initialize = False
            assert self.mesh.generation == self.generation == self.space.generation, 'must be'

        else:
            pass

    def _update(self):
        """"""
        if self._do_initialize:
            self._initialize()
        else:
            if self.mesh.generation == self.generation:
                assert self.space.generation == self.generation, f'must be'
            else:
                self.space._update()
                old_generation = self.generation
                assert self.space.generation == self.generation + 1, f"space generation must only be 1-gen ahead."
                current = self.space.generic
                self._previous = self._generic
                self._generic = MPI_Py_2D_Unstructured_Form(
                    current, self.degree,
                    dual_representation=self.is_dual_representation(),
                    **self._pAti_form
                )
                self._generic._name = self.name

                assert self._previous is not self._generic, 'must be'
                self._generation = self.space.generation
                assert self._generation == old_generation + 1, f'must be!'
                assert self.mesh.generation == self._generation, 'must be'

                assert self.mesh.generation == self.generation == self.space.generation, 'must be'

                # --- send constant properties to generic form ----
                self._generic.cf = self._cf_cache
                # =================================================

    # ------- key method ----------------------------------------------------------------------
    def evolve(self, amount_of_cochain=1):
        """take the cochain of `previous` and project them into `generic`."""
        link = self.mesh.link

        pre_f = self.previous
        cur_f = self.generic

        if pre_f is None:
            return
        else:
            pass

        old_csm = pre_f.space.basis_functions.csm(self.degree)
        new_csm = cur_f.space.basis_functions.csm(self.degree)

        pre_cochain = pre_f.cochain
        pre_cochain_times = list(pre_cochain._tcd.keys())
        cochain_times_2b_evolved = pre_cochain_times[-amount_of_cochain:]

        for t in cochain_times_2b_evolved:
            sour_cochain = pre_cochain[t].local
            dest_cochain: Dict = dict()

            for dest_index in link:
                source_indices = link[dest_index]
                if source_indices is None:
                    # cell is the same, cannot just pass it to the destination.
                    raw_cochain = sour_cochain[dest_index]
                    if dest_index in old_csm:
                        dest_cochain[dest_index] = old_csm[dest_index] @ raw_cochain
                    else:
                        dest_cochain[dest_index] = raw_cochain

                else:
                    if isinstance(source_indices, list):
                        # dest cell is coarser: get cochain from multiple smaller cell.
                        assert len(source_indices) > 1, f'Must be!'
                        local_source_cochains = dict()
                        for si in source_indices:
                            local_source_cochains[si] = sour_cochain[si]

                        cochain = self.space.coarsen(
                            self.degree, dest_index, source_indices, local_source_cochains
                        )
                    else:  # dest cell is more refined: get cochain from a bigger cell.
                        local_source_cochain = sour_cochain[source_indices]
                        cochain = self.space.refine(
                            self.degree, dest_index, source_indices, local_source_cochain
                        )
                    dest_cochain[dest_index] = cochain

            for index in dest_cochain:
                if index in new_csm:
                    dest_cochain[index] = new_csm[index] @ dest_cochain[index]
                else:
                    pass
            cur_f.cochain._set(t, dest_cochain)

    # ============================================================================================

    @property
    def cochain(self):
        """cochain; always use the current representative: generic"""
        return self.generic.cochain

    @property
    def boundary_integrate(self):
        """boundary_integrate; always use the current representative: generic"""
        return self.generic.boundary_integrate

    @property
    def numeric(self):
        return self._numeric

    @property
    def visualize(self):
        """visualize of current representative: generic."""
        return self.generic.visualize
