# -*- coding: utf-8 -*-
"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
@time: 11/26/2022 2:56 PM
"""

import sys

if './' not in sys.path:
    sys.path.append('./')

from src.tools.frozen import Frozen
from typing import Dict
from src.config import _mesh_partition_sym_repr, _mesh_partition_lin_repr


class BoundaryCondition(Frozen):
    """"""
    def __init__(self, mesh):
        """"""
        manifold = mesh.manifold
        self._boundary = manifold.boundary()
        self._manifold = manifold
        self._mesh = mesh
        _ = mesh.boundary()  # make sure the boundary mesh is made at the background.
        # keys: boundary_section_sym_repr, values: the given form on this boundary section.
        self._valid_bcs: Dict[str] = dict()
        self._freeze()

    def partition(self, *sym_reprs, config_name=None):
        """Define boundary sections by partition the mesh boundary into sections defined by `*sym_reprs`."""
        sub_manifolds = self._boundary.partition(*sym_reprs, config_name=config_name)
        for sub_manifold in sub_manifolds:
            sr0, sr1 = _mesh_partition_sym_repr
            if sub_manifold._covered_by_mesh is None:
                self._mesh.__class__(
                    sub_manifold,
                    sym_repr=sr0 + sub_manifold._sym_repr + sr1,
                    lin_repr=_mesh_partition_lin_repr + sub_manifold._pure_lin_repr,
                )  # it will be automatically saved to _global_meshes.
            else:
                pass

    def define_bc(self, bcs_dict):
        """"""
        assert isinstance(bcs_dict, dict), f"pls put boundary conditions in a dict whose keys are the boundary " \
                                           f"sections, and values are the B.C.s on the corresponding sections."
        for key in bcs_dict:
            assert key in self._boundary._sub_manifolds, f"boundary section: {key} is not defined yet."

            bcs = bcs_dict[key]

            if bcs.__class__.__name__ == 'Form':
                bcs = [bcs, ]
            else:
                pass
            for i, bc in enumerate(bcs):
                assert bc.__class__.__name__ == 'Form', f"{i}th BC: {bc} on Boundary Section {key} is not valid."
                if key not in self._valid_bcs:
                    self._valid_bcs[key] = list()
                else:
                    pass
                self._valid_bcs[key].append(bc)

    def __iter__(self):
        """go through all boundary section sym_repr that has valid BC on."""
        for boundary_sym_repr in self._valid_bcs:
            yield boundary_sym_repr

    def __getitem__(self, boundary_section_sym_repr):
        """Return the B.Cs on this boundary section."""
        assert boundary_section_sym_repr in self, f"no valid BC is defined on {boundary_section_sym_repr}."
        return self._valid_bcs[boundary_section_sym_repr]

    def __contains__(self, boundary_section_sym_repr):
        """Return True if there is valid BC defined on this boundary section."""
        return boundary_section_sym_repr in self._valid_bcs

    def __len__(self):
        """The number of boundary sections on which there is BC."""
        return len(self._valid_bcs)

    def _bc_text(self):
        """generate a text for print_representation of BC."""
        bc_text = '\n\nsubject to B.Cs:'
        for i, boundary_sym_repr in enumerate(self._valid_bcs):
            bc_text += f"\n({i}): Given "
            _ = list()
            for bc in self._valid_bcs[boundary_sym_repr]:
                _.append(rf"${bc._sym_repr}$")
            bc_text += ', '.join(_) + rf' on ${boundary_sym_repr}$;'

        all_partitions = self._boundary._partitions
        involved_partitions = list()
        for partition_key in all_partitions:
            involved = False
            for m in all_partitions[partition_key]:
                sym_repr = m._sym_repr
                if sym_repr in self:
                    involved = True
                    break
                else:
                    pass
            if involved and partition_key != '0':
                involved_partitions.append(partition_key)
        if len(involved_partitions) > 0:
            bc_text += '\nwhere'
            for i, ipk in enumerate(involved_partitions):
                ms = all_partitions[ipk]
                m_sym_repr_s = list()
                for m in ms:
                    m_sym_repr_s.append(m._sym_repr)
                m_sym_repr_s = r'\cup'.join(m_sym_repr_s)
                bc_text += ' $' + m_sym_repr_s + f" = {self._boundary._partitions['0'][0]._sym_repr} $"
                if i < len(involved_partitions) - 1:
                    bc_text += ', '
                else:
                    bc_text += '.'
        else:
            pass
        return bc_text


if __name__ == '__main__':
    # python src/bc.py

    import __init__ as ph

    samples = ph.samples

    oph = samples.pde_canonical_pH(3, 3)[0]

    wf = oph.test_with(oph.unknowns, sym_repr=[r'v^3', r'u^2'])
    wf = wf.derive.integration_by_parts('1-1')
    wf.pr()
