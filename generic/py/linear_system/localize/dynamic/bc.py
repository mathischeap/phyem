# -*- coding: utf-8 -*-
r"""
"""

from tools.frozen import Frozen

from src.manifold import Manifold

from src.wf.mp.linear_system_bc import _EssentialBoundaryCondition
from src.wf.mp.linear_system_bc import _NaturalBoundaryCondition


class Dynamic_Linear_System_BC(Frozen):
    """"""

    def __init__(self, dls, bc, base):
        """"""
        self._dls = dls
        self._abstract = bc
        self._parse(bc, base)
        self.___labels___ = None
        self._freeze()

    def _parse(self, abstract_mp_ls_bc, base):
        """"""
        manifolds = base['manifolds']
        self._valid_bcs = {}

        for abstract_boundary_section in abstract_mp_ls_bc:

            # now we try to find the msepy mesh for this abstract boundary section (manifold in fact).

            boundary_manifold = None
            for mani_sym in manifolds:
                boundary_manifold = manifolds[mani_sym]
                abstract_manifold = boundary_manifold.abstract
                if abstract_manifold._sym_repr == abstract_boundary_section:
                    break

                else:
                    pass

            assert boundary_manifold is not None, \
                f"we must have found a msepy manifold for the abstract boundary section."

            msepy_bcs_list = list()

            raw_bcs = abstract_mp_ls_bc[abstract_boundary_section]
            for raw_bc in raw_bcs:

                if raw_bc.__class__ is _EssentialBoundaryCondition:

                    msepy_bcs_list.append(
                        Essential_BoundaryCondition(base, boundary_manifold, raw_bc)
                    )

                elif raw_bc.__class__ is _NaturalBoundaryCondition:

                    msepy_bcs_list.append(
                        Natural_BoundaryCondition(base, boundary_manifold, raw_bc)
                    )

                else:
                    raise NotImplementedError(raw_bc)

            self._valid_bcs[abstract_boundary_section] = msepy_bcs_list

    def _bc_text(self):
        """"""
        return self._abstract._bc_text()

    def __len__(self):
        """The number of boundary sections on which there are BCs."""
        return len(self._abstract)

    def __iter__(self):
        """go through all boundary section sym_repr that has valid BC on."""
        for boundary_sym_repr in self._valid_bcs:
            yield boundary_sym_repr

    def __getitem__(self, boundary_section_sym_repr):
        """Return the B.Cs on this boundary section."""
        assert boundary_section_sym_repr in self, f"no valid BC is defined on {boundary_section_sym_repr}."
        return self._valid_bcs[boundary_section_sym_repr]

    def __len__(self):
        """How many valid boundary sections?"""
        return len(self._valid_bcs)

    def __contains__(self, boundary_section_sym_repr):
        """check if there is bc defined on `boundary_section_sym_repr`."""
        return boundary_section_sym_repr in self._valid_bcs

    def _labels(self):
        """label all boundary conditions."""
        if self.___labels___ is None:
            self.___labels___ = {}
            i = 0
            for boundary_section_repr in self._valid_bcs:
                bcs = self._valid_bcs[boundary_section_repr]
                for bc in bcs:
                    self.___labels___[i] = [
                        boundary_section_repr, bc
                    ]
                    i += 1

        return self.___labels___

    def list(self):
        """list all boundary conditions with labels."""
        for i in self._labels():
            print(i, self._labels()[i])

    def config(self, what):
        """Config a boundary condition to make it particular!"""

        if hasattr(what, 'abstract') and what.abstract.__class__ is Manifold:
            abstract_manifold = what.abstract
            manifold_sym_repr = abstract_manifold._sym_repr
            if manifold_sym_repr in self:

                assert len(self[manifold_sym_repr]) == 1, \
                    f"There are multiple bcs defined on {what}, " \
                    f"use other way to identify which bc you want to config"

                bc_2b_config = self[manifold_sym_repr][0]

            else:
                raise Exception(
                    f"No bc defined on {what}."
                )

        else:
            raise NotImplementedError(f"cannot recognize {what}.")

        return bc_2b_config


class BoundaryCondition(Frozen):
    """"""

    def __init__(self, base, boundary_manifold, raw_ls_bc):
        """"""
        self._base = base
        self._boundary_manifold = boundary_manifold
        self._raw_ls_bc = raw_ls_bc
        self._configuration = None  # the configuration
        self._num_application = 0   # for how many times this bc has taken effect
        self._freeze()

    def __call__(self, *args, **kwargs):
        """set default call to config!"""
        self.config(*args, **kwargs)

    def config(self, *args, **kwargs):
        """config the bc to be particular."""
        raise NotImplementedError()

    def __repr__(self):
        """repr"""
        return '<generic-mse-py DLS ' + self._raw_ls_bc.__repr__()[1:]


class Natural_BoundaryCondition(BoundaryCondition):
    """<tr star bf0 | tr bf1>"""

    def config(self, func_tr_star_bf0):
        """config the bc to be particular."""
        bf0 = self._raw_ls_bc._provided_root_form

        forms = self._base['forms']
        found_bf0 = None
        for msepy_form_sym_repr in forms:
            form = forms[msepy_form_sym_repr]
            if form.abstract is bf0:
                found_bf0 = form
                break
            else:
                pass

        assert found_bf0 is not None, f"we must have found a msepy form."

        self._configuration = func_tr_star_bf0


class Essential_BoundaryCondition(BoundaryCondition):
    """"""
    def __init__(self, base, boundary_manifold, raw_ls_bc):
        """"""
        super().__init__(base, boundary_manifold, raw_ls_bc)
        self._melt()
        self._cache1 = {
            'key': '',
            'find': tuple()
        }
        self._freeze()

    def config(self, exact_solution):
        """config the bc to be particular."""
        bf0 = self._raw_ls_bc._provided_root_form
        msepy_forms = self._base['forms']
        found_bf0 = None
        for form_sym_repr in msepy_forms:
            form = msepy_forms[form_sym_repr]
            if form.abstract is bf0:
                found_bf0 = form
                break
            else:
                pass

        assert found_bf0 is not None, f"we must have found a msepy form."
        self._configuration = exact_solution

    def apply(self, *args):
        """

        Parameters
        ----------
        args

        Returns
        -------

        """
        if self._configuration is None:
            return   # will make no changes to the *args
        else:
            pass

        from generic.py.linear_system.localize.dynamic.main import Dynamic_Linear_System

        # apply this essential boundary condition to dynamic linear system
        if len(args) == 4 and args[0].__class__ is Dynamic_Linear_System:
            dls, A, x, b = args
            self._apply_to_dynamic_linear_system(dls, A, x, b)

        else:
            raise NotImplementedError()

    def _find_dofs_and_cochains(self, unknowns, global_or_local='global'):
        """"""
        # --- decide the reduction time for the essential bc -----------------
        i = self._raw_ls_bc._i
        unknown = unknowns[i]
        from generic.py.vector.localize.static import Localize_Static_Vector_Cochain
        if unknown.__class__ is Localize_Static_Vector_Cochain:
            time = unknown._t
        else:
            raise NotImplementedError()
        # ====================================================================

        # below, we try to find the msepy mesh boundary section where the essential bc will apply.
        meshes = self._base['meshes']  # find mesh here! because only when we call it, the meshes are config-ed.
        found_boundary_section = None
        for mesh_sym_repr in meshes:
            msepy_mesh = meshes[mesh_sym_repr]
            msepy_manifold = msepy_mesh.manifold
            if msepy_manifold is self._boundary_manifold:
                found_boundary_section = msepy_mesh
                break
            else:
                pass
        assert found_boundary_section is not None, f"must found the mesh."

        # ------------ for different implementations ------------------------------------------------------
        from msehy.py._2d.mesh.main import MseHyPy2Mesh
        if found_boundary_section.__class__ is MseHyPy2Mesh:
            assert not found_boundary_section._is_mesh(), f"must represent a boundary section"
            found_boundary_section = found_boundary_section.representative.generic
        else:
            raise NotImplementedError()
        # =================================================================================================

        # below, we try to find the root-form this essential bc is for
        bf0 = self._raw_ls_bc._provided_root_form
        forms = self._base['forms']
        found_bf0 = None
        for msepy_form_sym_repr in forms:
            msepy_form = forms[msepy_form_sym_repr]
            if msepy_form.abstract is bf0:
                found_bf0 = msepy_form
                break
            else:
                pass
        assert found_bf0 is not None, f"we must have found a msepy form."

        # ------------ for different implementations ------------------------------------------------------
        from msehy.py._2d.form.main import MseHyPy2RootForm
        if found_bf0.__class__ is MseHyPy2RootForm:
            found_bf0 = found_bf0.generic
        else:
            raise NotImplementedError()
        # =================================================================================================

        # now, we  try to find the mesh-elements, local-dofs to be used.-----

        if len(found_boundary_section) == 0:  # this bc is valid on no faces. Just skip.
            return ()
        else:
            pass

        _0_elements = list()
        _1_local_dofs = list()
        for face_index in found_boundary_section:
            face = found_boundary_section[face_index]
            _0_elements.append(
                face._element_index
            )
            _1_local_dofs.append(
                face.find_local_dofs_of(found_bf0)
            )

        cf = self._configuration
        full_local_cochain = found_bf0.reduce(time, update_cochain=False, target=cf)

        # now, we try to pick the correct dofs from the full_local_cochain.
        _2_cochain = list()
        for i, e in enumerate(_0_elements):
            local_dofs = _1_local_dofs[i]
            _cochain = full_local_cochain[e][local_dofs]

            _2_cochain.append(
                _cochain
            )

        if global_or_local == 'local':
            assert len(_0_elements) == len(_1_local_dofs) == len(_2_cochain), f'safety check!'
            return _0_elements, _1_local_dofs, _2_cochain
        elif global_or_local == 'global':
            # below, we use _0_elements, _1_local_dofs to find global dofs and assemble _3_cochain accordingly.
            gm = found_bf0.cochain.gathering_matrix
            global_dofs = list()
            global_cochain = list()
            for i, e in enumerate(_0_elements):
                global_dofs.extend(gm[e][_1_local_dofs[i]])
                global_cochain.extend(_2_cochain[i])
            assert len(global_dofs) == len(global_cochain), f"must be the case."
            return global_dofs, global_cochain
        else:
            raise Exception()

    def _apply_to_dynamic_linear_system(self, dls, A, x, b):
        """"""
        assert dls.shape[0] == len(A) == len(x) == len(b), f"check!"

        _ = self._find_dofs_and_cochains(x, global_or_local='global')

        if _ == tuple():
            return
        else:
            global_dofs, global_cochain = _

        # customize the system
        if len(global_dofs) > 0:
            i = self._raw_ls_bc._i
            Ai = A[i]
            bi = b[i]
            for j, Aij in enumerate(Ai):

                if i == j:  # Aii, we are going to identify the diagonal of this matrix
                    Aij.customize.identify_diagonal(global_dofs)
                else:  # set the corresponding rows to be zero.
                    Aij.customize.set_zero(global_dofs)

            bi.customize.set_values(global_dofs, global_cochain)

        else:
            pass
