r"""Wrapper of methods for pass something one to another.

"""
from random import random

import numpy as np
from phyem.src.config import RANK, MASTER_RANK, COMM
from phyem.tools.frozen import Frozen
from phyem.msehtt.static.form.main import MseHttForm
from phyem.msehtt.tools.vector.static.global_gathered import MseHttGlobalVectorGathered


class MseHtt_MultiGrid_GreatMesh_Pass(Frozen):
    r""""""

    def __init__(self, tgm):
        r""""""
        self._tgm = tgm
        self._freeze()

    def cochain(self, ff, from_time, tf, to_time, complete_only=False):
        r"""
        Parameters
        ----------
        ff :
            from form
        from_time :
            from time
        tf :
            to form
        to_time :
            to time

        complete_only:
            If complete_only is True, we only accept complete global_dof_correspondence cochain passing.

        """
        if isinstance(ff, MseHttForm) and isinstance(tf, MseHttForm):
            ind, global_dof_correspondence = self._tgm.dof_correspondence.global_correspondence(tf, ff)

            if complete_only:
                assert ind == 'complete', \
                    (f"complete_only={complete_only}, so we only accept complete "
                     f"global_dof_correspondence for cochain passing.")
            else:
                pass

            if global_dof_correspondence is None:  # we find no correspondence, then do a general projection
                assert not complete_only, f"We only accept complete global_dof_correspondence cochain passing."
                assert ind == 'empty', f"must be empty."
                tsp = ff.numeric.tsp(from_time)
                tf[to_time].reduce(tsp[from_time])

            elif isinstance(ind, str) and ind == 'incomplete':
                assert isinstance(global_dof_correspondence, dict), f"incomplete dof cor information must be returned."
                assert not complete_only, f"We only accept complete global_dof_correspondence cochain passing."
                tsp = ff.numeric.tsp(from_time)
                tf[to_time].reduce(tsp[from_time])

            elif isinstance(global_dof_correspondence, str) and global_dof_correspondence == 'same':
                assert ind == 'complete', f"must be complete."
                raise NotImplementedError(f"complete and SAME")

            elif isinstance(global_dof_correspondence, dict) and isinstance(ind, str) and ind == 'complete':
                f_gm = ff.cochain.gathering_matrix
                ffc = ff[from_time].cochain
                ffc_vec = np.zeros(f_gm.num_global_dofs, dtype=float)

                for e in ffc:
                    ffc_vec[f_gm[e]] = ffc[e]

                t_gm = tf.cochain.gathering_matrix
                tf_cochain_data_dict = {}
                for e in t_gm:
                    tf_element_dofs = t_gm[e]
                    tf_element_cochain = np.zeros(len(tf_element_dofs))
                    for i, tf_dof in enumerate(tf_element_dofs):
                        # assert tf_dof in global_dof_correspondence, f"must be since the global dof cor is complete."
                        # ff_dofs = global_dof_correspondence[tf_dof]
                        # value = 0
                        # for ff_dof in ff_dofs:
                        #     value += ffc_vec[ff_dof]
                        # value = sum(ffc_vec[ff_dofs])
                        tf_element_cochain[i] = sum(ffc_vec[global_dof_correspondence[tf_dof]])
                    tf_cochain_data_dict[e] = tf_element_cochain

                tf[to_time].cochain = tf_cochain_data_dict

            else:
                raise NotImplementedError(ind)

        else:
            raise NotImplementedError()

    def vector_through_cochain(self, vec, ff, tf, complete_only=False):
        r"""We pass a vector on one level to another level. This vector is considered as a cochain of the
        form `from_base_form`. It is passed to the cochain of the `to_base_form`. And this new cochain is
        regarded as the output vector.

        So, basically, we consider a vector as a cochain, and we pass it in-between meshes on different levels.
        And the principle is that the error between the reconstructions is minimized.

        Parameters
        ----------
        vec : {MseHttGlobalVectorGathered, }
        ff
        tf
        complete_only

        Returns
        -------
        to_vec : MseHttGlobalVectorGathered

        """
        if isinstance(ff, MseHttForm) and isinstance(tf, MseHttForm):
            ind, global_dof_correspondence = self._tgm.dof_correspondence.global_correspondence(tf, ff)

            if complete_only:
                assert ind == 'complete', \
                    (f"complete_only={complete_only}, so we only accept complete "
                     f"global_dof_correspondence for cochain passing.")
            else:
                pass

            if global_dof_correspondence is None:  # we find no correspondence, then do a general projection
                assert not complete_only, f"We only accept complete global_dof_correspondence cochain passing."
                assert ind == 'empty', f"must be empty."

                if RANK == MASTER_RANK:
                    t = - random()
                else:
                    t = None
                t = COMM.bcast(t, root=MASTER_RANK)

                ff[t].cochain = vec
                tsp = ff.numeric.tsp(t)
                tf[t].reduce(tsp[t])

                to_vec = tf[t].cochain.gathered_global_vector

                ff.cochain.clean(t=t)
                tf.cochain.clean(t=t)

                return to_vec

            elif isinstance(ind, str) and ind == 'incomplete':  # use the same strategy as None global-dof-cor.
                assert isinstance(global_dof_correspondence, dict), f"incomplete dof cor information must be returned."
                assert not complete_only, f"We only accept complete global_dof_correspondence cochain passing."

                if RANK == MASTER_RANK:
                    t = - random()
                else:
                    t = None
                t = COMM.bcast(t, root=MASTER_RANK)

                ff[t].cochain = vec
                tsp = ff.numeric.tsp(t)
                tf[t].reduce(tsp[t])

                to_vec = tf[t].cochain.gathered_global_vector

                ff.cochain.clean(t=t)
                tf.cochain.clean(t=t)

                return to_vec

            elif isinstance(global_dof_correspondence, str) and global_dof_correspondence == 'same':
                assert ind == 'complete', f"must be complete."
                raise NotImplementedError(f"complete and SAME")

            elif isinstance(global_dof_correspondence, dict) and isinstance(ind, str) and ind == 'complete':
                f_gm = ff.cochain.gathering_matrix
                t_gm = tf.cochain.gathering_matrix

                assert isinstance(vec, MseHttGlobalVectorGathered), f"vec must be a gathered global vector."
                assert vec._gm == f_gm, f"gathered global vector must have a correct gathering matrix."
                assert vec.shape == (f_gm.num_global_dofs, ), f"shape of vec does not match the from-form."
                vec = vec.V

                tf_cochain_data_dict = {}
                for e in t_gm:
                    tf_element_dofs = t_gm[e]
                    tf_element_cochain = np.zeros(len(tf_element_dofs))
                    for i, tf_dof in enumerate(tf_element_dofs):
                        # assert tf_dof in global_dof_correspondence, f"must be since the global dof cor is complete."
                        # ff_dofs = global_dof_correspondence[tf_dof]
                        # value = 0
                        # for ff_dof in ff_dofs:
                        #     value += vec[ff_dof]
                        # tf_element_cochain[i] = value
                        tf_element_cochain[i] = sum(vec[global_dof_correspondence[tf_dof]])
                    tf_cochain_data_dict[e] = tf_element_cochain

                to_vec = t_gm.assemble(tf_cochain_data_dict, mode='replace')
                # assemble of GM returns the same full vector in all ranks.
                return MseHttGlobalVectorGathered(to_vec, gm=t_gm)

            else:
                raise NotImplementedError(ind)

        else:
            raise NotImplementedError()
