# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msepy.manifold.main import MsePyManifold

from msepy.form.static import MsePyRootFormStaticCopy
from msepy.form.cochain.vector.static import MsePyRootFormStaticCochainVector

from msepy.form.cf import MsePyContinuousForm
from msepy.mesh.boundary_section.main import MsePyBoundarySectionMesh

from src.wf.mp.linear_system_bc import _EssentialBoundaryCondition
from src.wf.mp.linear_system_bc import _NaturalBoundaryCondition


class MsePyDynamicLinearSystemBoundaryCondition(Frozen):
    """"""

    def __init__(self, dynamic_ls, abstract_mp_ls_bc, msepy_base):
        self._dls = dynamic_ls
        self._abstract = abstract_mp_ls_bc
        self._parse(abstract_mp_ls_bc, msepy_base)
        self.___labels___ = None
        self._freeze()

    def _parse(self, abstract_mp_ls_bc, msepy_base):
        """"""
        msepy_manifolds = msepy_base['manifolds']
        self._valid_bcs = {}

        for abstract_boundary_section in abstract_mp_ls_bc:

            # now we try to find the msepy mesh for this abstract boundary section (manifold in fact).

            msepy_boundary_manifold = None
            for mani_sym in msepy_manifolds:
                msepy_boundary_manifold = msepy_manifolds[mani_sym]
                abstract_manifold = msepy_boundary_manifold.abstract
                if abstract_manifold._sym_repr == abstract_boundary_section:
                    break

                else:
                    pass

            assert msepy_boundary_manifold is not None, \
                f"we must have found a msepy manifold for the abstract boundary section."

            msepy_bcs_list = list()

            raw_bcs = abstract_mp_ls_bc[abstract_boundary_section]
            for raw_bc in raw_bcs:

                if raw_bc.__class__ is _EssentialBoundaryCondition:

                    msepy_bcs_list.append(
                        MsePyDLSEssentialBoundaryCondition(msepy_boundary_manifold, raw_bc)
                    )

                elif raw_bc.__class__ is _NaturalBoundaryCondition:

                    msepy_bcs_list.append(
                        MsePyDLSNaturalBoundaryCondition(msepy_boundary_manifold, raw_bc)
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

        if isinstance(what, MsePyManifold):
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


class MsePyDLSBoundaryCondition(Frozen):
    """"""

    def __init__(self, msepy_boundary_manifold, raw_ls_bc):
        """"""
        self._msepy_boundary_manifold = msepy_boundary_manifold
        self._raw_ls_bc = raw_ls_bc
        self._category = None       # configuration category
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
        return '<MsePy DLS ' + self._raw_ls_bc.__repr__()[1:]


class MsePyDLSNaturalBoundaryCondition(MsePyDLSBoundaryCondition):
    """<tr star bf0 | tr bf1>"""

    def config(self, *args, category=0):
        """config the bc to be particular."""
        bf0 = self._raw_ls_bc._provided_root_form
        from msepy.main import base
        msepy_forms = base['forms']
        found_msepy_bf0 = None
        for msepy_form_sym_repr in msepy_forms:
            msepy_form = msepy_forms[msepy_form_sym_repr]
            if msepy_form.abstract is bf0:
                found_msepy_bf0 = msepy_form
                break
            else:
                pass

        assert found_msepy_bf0 is not None, f"we must have found a msepy form."
        assert self._category is None, f"bc config-ed, change the configuration is dangerous!"

        # --------------------- 0: general vc ------------------------------------
        if category in (0, 'general_vc'):
            # we provide ONE general vc object as ``tr star bf0`` (NOT bf0) everywhere.

            assert len(args) == 1, f"for ``general_vc`` type configuration, only provide a vc object."

            configuration = args[0]
            assert configuration._is_time_space_func(), \
                f"for ``general_vc`` type configuration, only accept a vc object."

            self._category = 'general_vc'
            self._configuration = configuration

        # ---------------- else: not implemented ---------------------------------
        else:
            raise Exception(f"type = {type} is not understandable.")


class MsePyDLSEssentialBoundaryCondition(MsePyDLSBoundaryCondition):
    """"""
    def __init__(self, msepy_boundary_manifold, raw_ls_bc):
        """"""
        super().__init__(msepy_boundary_manifold, raw_ls_bc)
        self._melt()
        self._cache1 = {
            'key': '',
            'find': tuple()
        }
        self._freeze()

    def config(self, *args):
        """config the bc to be particular."""
        from msepy.main import base
        bf0 = self._raw_ls_bc._provided_root_form
        msepy_forms = base['forms']
        found_msepy_bf0 = None
        for msepy_form_sym_repr in msepy_forms:
            msepy_form = msepy_forms[msepy_form_sym_repr]
            if msepy_form.abstract is bf0:
                found_msepy_bf0 = msepy_form
                break
            else:
                pass

        assert found_msepy_bf0 is not None, f"we must have found a msepy form."
        assert self._category is None, f"bc config-ed, change the configuration is dangerous!"

        # ---------------- 0: general vc -------------------------------
        if len(args) == 1 and args[0].__class__ is MsePyContinuousForm:
            # we provide ONE general cf as the bc. Reducing this cf to boundary dofs leads to given boundary dofs.
            self._category = 'cf'
            self._configuration = args[0]
        elif len(args) == 1 and hasattr(args[0], f"_is_time_space_func"):
            self._category = 'general_vc'
            self._configuration = args[0]

        # ---------------- else: not implemented -----------------------
        else:
            raise Exception(f"configuration args not understandable.")

    def apply(self, *args):
        """

        Parameters
        ----------
        args

        Returns
        -------

        """
        from msepy.tools.linear_system.dynamic.main import MsePyDynamicLinearSystem
        from msepy.tools.nonlinear_system.static.local import MsePyStaticLocalNonLinearSystem

        # apply this essential boundary condition to dynamic linear system
        if len(args) == 4 and args[0].__class__ is MsePyDynamicLinearSystem:
            dls, A, x, b = args
            self._apply_to_dynamic_linear_system(dls, A, x, b)
        # apply this essential boundary condition to static local nonlinear system
        elif len(args) == 1 and args[0].__class__ is MsePyStaticLocalNonLinearSystem:
            nls = args[0]
            self._apply_to_static_local_nonlinear_system(nls)
        else:
            raise NotImplementedError()

    def _find_dofs_and_cochains(self, unknowns, global_or_local='global'):
        """"""
        i = self._raw_ls_bc._i
        unknown = unknowns[i]

        key = unknown.__repr__() + global_or_local

        if key == self._cache1['key']:
            return self._cache1['find']
        else:
            self._cache1['key'] = key

        # below, we try to find the msepy mesh boundary section where the essential bc will apply.
        from msepy.main import base
        meshes = base['meshes']  # find mesh here! because only when we call it, the meshes are config-ed.
        found_msepy_boundary_section_mesh = None
        for mesh_sym_repr in meshes:
            msepy_mesh = meshes[mesh_sym_repr]
            msepy_manifold = msepy_mesh.manifold
            if msepy_manifold is self._msepy_boundary_manifold:
                found_msepy_boundary_section_mesh = msepy_mesh
                break
            else:
                pass
        assert found_msepy_boundary_section_mesh is not None, f"must found the mesh."
        assert found_msepy_boundary_section_mesh.__class__ is MsePyBoundarySectionMesh, \
            f"we must have found a mesh boundary section."

        # below, we try to find the msepy root-form this essential bc is for
        bf0 = self._raw_ls_bc._provided_root_form
        msepy_forms = base['forms']
        found_msepy_bf0 = None
        for msepy_form_sym_repr in msepy_forms:
            msepy_form = msepy_forms[msepy_form_sym_repr]
            if msepy_form.abstract is bf0:
                found_msepy_bf0 = msepy_form
                break
            else:
                pass
        assert found_msepy_bf0 is not None, f"we must have found a msepy form."

        # now, we  try to find the mesh-elements, local-dofs to be used.-----

        faces = found_msepy_boundary_section_mesh.faces
        if len(faces) == 0:  # this bc is valid on no faces. Just skip.
            self._cache1['find'] = tuple()
            return self._cache1['find']
        else:
            pass

        _0_elements = list()
        _1_local_dofs = list()
        for face_id in faces:
            face = faces[face_id]
            _0_elements.append(
                face._element
            )
            _1_local_dofs.append(
                face.find_corresponding_local_dofs_of(found_msepy_bf0)
            )

        # below, we try to find the coefficients to be used. This is the most important part.

        if unknown.__class__ is MsePyRootFormStaticCopy:
            time = unknown._t
            if found_msepy_bf0._is_base():  # unknown._f must be the base form.
                assert found_msepy_bf0 is unknown._f, 'safety check!'
            else:
                assert found_msepy_bf0._base is unknown._f, 'safety check!'

        elif unknown.__class__ is MsePyRootFormStaticCochainVector:
            time = unknown._time
        else:
            raise Exception()

        if self._category == 'cf':
            cf = self._configuration
            full_local_cochain = found_msepy_bf0.reduce(time, update_cochain=False, target=cf)

        elif self._category == 'general_vc':
            vc = self._configuration
            full_local_cochain = found_msepy_bf0.reduce(time, update_cochain=False, target=vc)

        else:
            raise NotImplementedError(f"not implemented for bc config category = {self._category}")

        # now, we try to pick the correct dofs from the full_local_cochain.
        _2_cochain = list()
        for i, e in enumerate(_0_elements):
            local_dofs = _1_local_dofs[i]
            _cochain = full_local_cochain[e, local_dofs]

            _2_cochain.append(
                _cochain
            )

        if global_or_local == 'local':
            assert len(_0_elements) == len(_1_local_dofs) == len(_2_cochain), f'safety check!'
            self._cache1['find'] = _0_elements, _1_local_dofs, _2_cochain
        elif global_or_local == 'global':
            # below, we use _0_elements, _1_local_dofs to find global dofs and assemble _3_cochain accordingly.
            gm = found_msepy_bf0.cochain.gathering_matrix
            global_dofs = list()
            global_cochain = list()
            for i, e in enumerate(_0_elements):
                global_dofs.extend(gm[e][_1_local_dofs[i]])
                global_cochain.extend(_2_cochain[i])

            assert len(global_dofs) == len(global_cochain), f"must be the case."
            self._cache1['find'] = global_dofs, global_cochain

        else:
            raise Exception()

        return self._cache1['find']

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

    def _apply_to_static_local_nonlinear_system(self, nls):
        """"""
        _ = self._find_dofs_and_cochains(nls.unknowns, global_or_local='local')
        if _ == tuple():
            return
        else:
            elements, local_dofs, local_cochain = _
        if len(elements) > 0:
            i = self._raw_ls_bc._i
            nls.customize.set_x0_from_local_dofs(i, elements, local_dofs, local_cochain)
            nls.customize.set_no_evaluation_of_local_dofs(i, elements, local_dofs)
        else:
            pass
