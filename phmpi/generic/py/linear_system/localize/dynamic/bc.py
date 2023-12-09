# -*- coding: utf-8 -*-
r"""
"""

from src.config import RANK, MASTER_RANK, COMM

from src.wf.mp.linear_system_bc import _EssentialBoundaryCondition
from src.wf.mp.linear_system_bc import _NaturalBoundaryCondition

from legacy.generic.py.linear_system.localize.dynamic.bc import (Dynamic_Linear_System_BC,
                                                                 Essential_BoundaryCondition,
                                                                 Natural_BoundaryCondition)

from phmpi.generic.py.vector.localize.static import MPI_PY_Localize_Static_Vector_Cochain


class MPI_PY_Dynamic_Linear_System_BC(Dynamic_Linear_System_BC):
    """"""
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
                        MPI_PY_Essential_BoundaryCondition(base, boundary_manifold, raw_bc)
                    )

                elif raw_bc.__class__ is _NaturalBoundaryCondition:

                    msepy_bcs_list.append(
                        Natural_BoundaryCondition(base, boundary_manifold, raw_bc)
                    )

                else:
                    raise NotImplementedError(raw_bc)

            self._valid_bcs[abstract_boundary_section] = msepy_bcs_list

    def list(self):
        """list all boundary conditions with labels."""
        labels = self._labels()
        for i in labels:
            if RANK == MASTER_RANK:
                print(i, labels[i])
            else:
                pass


class MPI_PY_Essential_BoundaryCondition(Essential_BoundaryCondition):
    """"""

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

        # apply this essential boundary condition to dynamic linear system
        from phmpi.generic.py.linear_system.localize.dynamic.main import MPI_PY_Dynamic_Linear_System

        if len(args) == 4 and args[0].__class__ is MPI_PY_Dynamic_Linear_System:
            dls, A, x, b = args
            self._apply_to_dynamic_linear_system(dls, A, x, b)

        else:
            raise NotImplementedError()

    def _find_dofs_and_cochains(self, unknowns, global_or_local='global'):
        """"""
        # --- decide the reduction time for the essential bc -----------------
        i = self._raw_ls_bc._i
        unknown = unknowns[i]

        if unknown.__class__ is MPI_PY_Localize_Static_Vector_Cochain:
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
        from phmpi.msehy.py._2d.mesh.main import MPI_MseHy_Py2_Mesh
        if found_boundary_section.__class__ is MPI_MseHy_Py2_Mesh:
            assert not found_boundary_section._is_mesh(), f"must represent a boundary section"
            found_boundary_section = found_boundary_section.generic
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
        from phmpi.msehy.py._2d.form.main import MPI_MseHy_Py2_Form
        if found_bf0.__class__ is MPI_MseHy_Py2_Form:
            found_bf0 = found_bf0.generic
        else:
            raise NotImplementedError()
        # =================================================================================================

        # now, we  try to find the mesh-elements, local-dofs to be used.-----
        if found_boundary_section.num_total_covered_faces == 0:  # this bc is valid on no faces. Just skip.
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

            # --- we need to gather them to make them complete in all ranks

            global_dofs = COMM.gather(global_dofs, root=MASTER_RANK)
            global_cochain = COMM.gather(global_cochain, root=MASTER_RANK)
            if RANK == MASTER_RANK:
                GLOBAL_DOFS = list()
                GLOBAL_COCHAINS = list()
                for _d, _c in zip(global_dofs, global_cochain):
                    GLOBAL_DOFS.extend(_d)
                    GLOBAL_COCHAINS.extend(_c)
                global_dofs = GLOBAL_DOFS
                global_cochain = GLOBAL_COCHAINS
            else:
                pass

            global_dofs = COMM.bcast(global_dofs, root=MASTER_RANK)
            global_cochain = COMM.bcast(global_cochain, root=MASTER_RANK)

            return global_dofs, global_cochain
        else:
            raise Exception()
