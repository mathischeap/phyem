# -*- coding: utf-8 -*-
r"""
"""

from phyem.tools.frozen import Frozen


class MsePyStaticNonlinearSystemCustomize(Frozen):
    """"""

    def __init__(self, nls):
        """"""
        self._nls = nls
        self._customizations = list()
        self._freeze()

    @property
    def customization(self):
        """customization."""
        return self._customizations

    def clear(self):
        """"""
        self._customizations = list()

    def set_no_evaluation(self, dof):
        """Let the nonlinear system do not affect the value of global #r dof.

        So cochain of `dof` will be equal to its x0 value (the initial value (or initial guess)).
        """
        self._customizations.append(
            ('set_no_evaluation', dof)
        )

    def set_dof(self, dof, v):
        """Set the global dof #dof to be v.

        Parameters
        ----------
        dof
        v

        Returns
        -------

        """
        elements_local_dofs = self._nls._global_row_gm._find_elements_and_local_indices_of_dofs(dof)
        elements, local_dofs = elements_local_dofs[list(elements_local_dofs.keys())[0]]

        col_gms = self._nls._col_gms
        in_known_number = -1
        unknown_local_dofs = list()
        acc = 0
        for j, gmj in enumerate(col_gms):
            ACC = acc + gmj.shape[1]
            if all([_ < ACC for _ in local_dofs]):
                in_known_number = j
                for local_dof in local_dofs:
                    unknown_local_dofs.append(
                        local_dof - acc
                    )
                break
            else:
                pass
            acc += gmj.shape[1]

        assert in_known_number != -1
        self.set_x0_from_local_dofs(
            in_known_number, elements, unknown_local_dofs, [v for _ in unknown_local_dofs]
        )
        self.set_no_evaluation(dof)

    def set_x0_from_local_dofs(self, i, elements, local_dofs, local_values):
        """Set x0 for the #`i` unknown at its local positions (indicated by `elements` and `local_dofs`)
        to be `local values`
        """
        self._customizations.append(
            ('set_x0_from_local_dofs', [i, elements, local_dofs, local_values])
        )

    def set_no_evaluation_of_local_dofs(self, i, elements, local_dofs):
        """Let the local dofs (indicating by `elements` and `local_dofs`) of the #`i` unknown do not change.
        """
        unknowns = self._nls.unknowns
        local_dofs_base = 0
        if i > 0:
            for j in range(i):
                local_dofs_base += unknowns[j]._f.cochain.gathering_matrix.num_local_dofs

            overall_local_dofs = list()   # consider the local system as one matrix

            for ld in local_dofs:
                overall_local_dofs.append(
                    ld + local_dofs_base
                )

        else:
            overall_local_dofs = local_dofs

        self._customizations.append(
            ('set_no_evaluation_for_overall_local_dofs', [elements, overall_local_dofs])
        )

    def _apply_essential_BC_to_linear_part(
            self,
            i, j,
            boundary_section,
            bc_function, t=None
    ):
        """Apply essential-like B.C. to the block[i][j] of the linear part of the nonlinear system.

        We change the dofs on `faces` of `boundary_section` to the cochain reduced from
        `bc_function` at time `t`.

        Parameters
        ----------
        i
        j
        boundary_section
        bc_function
        t

        Returns
        -------

        """
        if i == j:
            self._apply_diagonal_essential_BC_to_linear_part(
                i,
                boundary_section,
                bc_function, t=t
            )
        else:
            pass

        sfi = self._nls.unknowns[i]   # it is an essential B.C. for this form (static copy)
        sfj = self._nls.unknowns[j]
        fi = sfi._f
        fj = sfj._f

        if t is None:  # we get the time from the unknown.
            t = sfj._t
        else:
            if t != sfj._t:
                print("warning!, time differs from unknown time.")
            else:
                pass

        cochain = fj.reduce(t, update_cochain=False, target=bc_function)

        # now we parse the boundary section
        the_msepy_boundary_section = None   # to store the found msepy boundary section
        from msepy.manifold.main import MsePyManifold
        from msepy.main import base
        if boundary_section.__class__ is MsePyManifold:

            meshes = base['meshes']
            for sym in meshes:
                mesh = meshes[sym]
                if mesh.abstract.manifold is boundary_section.abstract:
                    the_msepy_boundary_section = mesh
                    break

        else:
            raise NotImplementedError(f"do not understand boundary section: {boundary_section}.")

        assert the_msepy_boundary_section is not None, \
            f"Cannot find a MsePyBoundarySection instance from {boundary_section}."

        # parse the global-dofs and global-cochains to be applied ----

        dofs_i = the_msepy_boundary_section.find_boundary_objects(
            fi, 'gathering_matrix'
        )
        dofs_j, cochain = the_msepy_boundary_section.find_boundary_objects(
            fj, 'gathering_matrix', cochain
        )

        assert len(dofs_i) == len(dofs_j) == len(cochain), f'must be!'

        # --- find the related blocks in A, b of the linear part -------
        A = self._nls._A
        b = self._nls._b
        bi = b[i]             # the right-hand vector to be customized
        Aij = A[i][j]         # the left-hand matrix block to identify
        Ai_ = list()          # the left-hand matrix blocks to zero
        for k, A__ in enumerate(A[i]):
            if k != j:
                Ai_.append(A__)

        # ----- make the changes --------------------------------------
        for aij in Ai_:  # zeros all rows in Blocks Ai_ (not in Aij).
            aij.customize.set_zero(dofs_i)

        Aij.customize.set_values_and_zero_rest(dofs_i, dofs_j, 1)
        bi.customize.set_values(dofs_i, cochain)

    def _apply_diagonal_essential_BC_to_linear_part(
            self,
            i,
            boundary_section,
            bc_function, t=None
    ):
        """

        Parameters
        ----------
        i
        boundary_section
        bc_function
        t

        Returns
        -------

        """
        sf = self._nls.unknowns[i]  # it is an essential B.C. for this form (static copy)
        f = sf._f

        if t is None:  # we get the time from the unknown.
            t = sf._t
        else:
            if t != sf._t:
                print("warning!, time differs from unknown time.")
            else:
                pass

        cochain = f.reduce(t, update_cochain=False, target=bc_function)

        # now we parse the boundary section
        the_msepy_boundary_section = None   # to store the found msepy boundary section
        from msepy.manifold.main import MsePyManifold
        from msepy.main import base
        if boundary_section.__class__ is MsePyManifold:

            meshes = base['meshes']
            for sym in meshes:
                mesh = meshes[sym]
                if mesh.abstract.manifold is boundary_section.abstract:
                    the_msepy_boundary_section = mesh
                    break

        else:
            raise NotImplementedError(f"do not understand boundary section: {boundary_section}.")

        dofs, cochain = the_msepy_boundary_section.find_boundary_objects(
            f, 'gathering_matrix', cochain
        )

        assert len(dofs) == len(cochain), f'must be!'

        # --- find the related blocks in A, b of the linear part -------
        A = self._nls._A
        b = self._nls._b
        bi = b[i]             # the right-hand vector to be customized
        Aii = A[i][i]         # the diagonal left-hand matrix block to identify
        Ai_ = list()          # the off-diagonal left-hand matrix blocks to zero
        for k, A__ in enumerate(A[i]):
            if k != i:
                Ai_.append(A__)

        # ----- make the changes --------------------------------------
        for aij in Ai_:  # zeros all rows in Blocks Ai_ (not in Aij).
            aij.customize.set_zero(dofs)

        Aii.customize.identify_diagonal(dofs)
        bi.customize.set_values(dofs, cochain)
