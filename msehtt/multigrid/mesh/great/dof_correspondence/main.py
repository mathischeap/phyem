# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen
from phyem.msehtt.static.form.main import MseHttForm
from phyem.msehtt.multigrid.mesh.great.dof_correspondence.global_ import _globalDofCorresponding__msehtt_static_form_


class MseHtt_MultiGrid_GreatMesh_DofCorrespondence(Frozen):
    r"""We wrap all methods of finding dof-correspondence here in this class.

    Logically, this class should be for MG-form. But, put it here is better for implementation. For example,
    If we receive two msehtt-static forms, and we want to know their dof-correspondence, then if we call
    the method from their MG-form, we need to firstly find out which MG-form their belong to. And furthermore,
    if they are from different MG-forms (i.e. they are levels of different MG-forms), from which MG-form we should
    call the method? All these problems will complicate the implementation. However, this is ONE object which is
    unique everywhere, it is the MG-great-mesh. If we call the methods from it, we can bypass all these issues.
    And that is why this class is like a property of the MG-great-mesh class.
    """

    def __init__(self, tgm):
        r""""""
        self._tgm = tgm
        self._global_correspondence_cache_ = {}
        self._freeze()

    def global_correspondence(self, f0, f1):
        r"""
        CASE 1:
        This returns a dict, for example, `global_dof_cor`.

        Then `global_dof_cor[i]` is for the global dof #i of f0. And let's say, if `global_dof_cor[i] = A`.
        Then A is a list of all global dof indices of f1 that is completely on the global dof #i of f0.

        So, basically, we are trying to find the global dofs of f1 that are on each global dof of f0.

        CASE 2:
        When the two forms are actually in the same space (must also on the same level mesh), we return 'same'.

        CASE 3:
        When we find no correspondence, return None.

        """
        key = f0.__repr__() + '<G@C>' + f1.__repr__()
        if key in self._global_correspondence_cache_:
            return self._global_correspondence_cache_[key]
        else:
            pass

        if isinstance(f0, MseHttForm) and isinstance(f1, MseHttForm):
            ind, cor = _globalDofCorresponding__msehtt_static_form_(self._tgm, f0, f1)
        else:
            raise NotImplementedError()

        if cor is None:
            RETURN = ('empty', None)
        elif isinstance(cor, dict) and len(cor) == 0:
            RETURN = ('empty', None)
        elif isinstance(cor, str) and cor == 'same':
            RETURN = ('complete', 'same')
        elif isinstance(ind, str) and ind == 'incomplete':
            RETURN = ('incomplete', cor)
        else:
            assert ind == 'complete' and isinstance(cor, dict), f"If find a correspondence, let it be a dict!"
            if isinstance(f0, MseHttForm):
                gm = f0.cochain.gathering_matrix
                for e in gm:
                    for f0_global_dof in gm[e]:
                        assert f0_global_dof in cor, \
                            f"A correspondence must be complete. Now, we miss global dof #{f0_global_dof}"

            else:
                raise NotImplementedError()

            RETURN = ('complete', cor)

        self._global_correspondence_cache_[key] = RETURN
        return RETURN
