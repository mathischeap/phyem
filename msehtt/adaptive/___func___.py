# -*- coding: utf-8 -*-
r"""
"""
from phyem.src.config import MASTER_RANK, RANK, COMM
from phyem.msehtt.static.mesh.partial.main import MseHttMeshPartial


def ___func_renew___(new_tgm, base):
    r""""""
    # --------- renew meshes ------------------------------------
    ___config_meshes___ = base['___config_meshes___']
    for mesh_sym_repr in ___config_meshes___:
        # to make sure meshes are initialized in the sequence as they configured.
        assert mesh_sym_repr in base['meshes']
        mesh = base['meshes'][mesh_sym_repr]
        if mesh._including_ is None:
            pass
        else:
            mesh.___renew___(new_tgm)

    COMM.barrier()

    # ---------- renew spaces -----------------------------------
    for ab_sp_sym_repr in base['spaces']:
        space = base['spaces'][ab_sp_sym_repr]
        ab_sp = space._abstract_space

        if ab_sp.orientation == 'unknown':
            # These spaces are probably not for root-forms, skipping is OK.
            pass
        else:
            space.___renew___()

            the_msehtt_partial_mesh = None
            for mesh_repr in base['meshes']:
                if mesh_repr == space._abstract_space.mesh._sym_repr:
                    assert base['meshes'][mesh_repr]._total_generation_ == space._total_generation_, \
                        f"mesh: {mesh_repr} is not configured or outdated."
                    the_msehtt_partial_mesh = base['meshes'][mesh_repr].current
                    break
                else:
                    pass

            assert isinstance(the_msehtt_partial_mesh, MseHttMeshPartial), f"we must have found a msehtt partial mesh!"
            space.current._tpm = the_msehtt_partial_mesh

    COMM.barrier()

    # ---------- renew forms ------------------------------------
    for pur_lin_repr in base['forms']:
        form = base['forms'][pur_lin_repr]
        abs_rf = form._abstract_form
        assert pur_lin_repr == abs_rf._pure_lin_repr
        if abs_rf._pAti_form['base_form'] is None:  # this is not a root-form at a particular time-instant.
            form.___renew___()
        else:
            pass

    COMM.barrier()

    for pur_lin_repr in base['forms']:  # then do it for all root-forms at particular time instant
        form = base['forms'][pur_lin_repr]
        abs_rf = form._abstract_form

        if abs_rf._pAti_form['base_form'] is None:
            pass
        else:    # this is a root-form at a particular time-instant.
            base_form = abs_rf._pAti_form['base_form']
            ats = abs_rf._pAti_form['ats']
            ati = abs_rf._pAti_form['ati']

            particular_base_form = base['forms'][base_form._pure_lin_repr].current
            form.___renew___()
            form.current._pAti_form['base_form'] = particular_base_form
            form.current._pAti_form['ats'] = ats
            form.current._pAti_form['ati'] = ati

            assert pur_lin_repr not in particular_base_form._ats_particular_forms
            particular_base_form._ats_particular_forms[pur_lin_repr] = form.current

    for pur_lin_repr in base['forms']:
        form = base['forms'][pur_lin_repr]
        assert form.current.degree is not None, \
            f"msehtt form must have a degree, an abstract root of no degree cannot be implemented."

    COMM.barrier()


def ___base_tgm___(base):
    r""""""
    _BASE_ = base['_BASE_']
    if _BASE_ is None:
        _BASE_ = {}
        TGM = base['the_great_mesh']
        if RANK == MASTER_RANK:
            _BASE_['element_type_dict'] = TGM._global_element_type_dict
            _BASE_['element_map_dict'] = TGM._global_element_map_dict
        else:
            _BASE_['element_type_dict'] = None
            _BASE_['element_map_dict'] = None
            _BASE_['element_parameter_dict'] = None

        _global_element_map_dict = {}
        for e in TGM.elements:
            element = TGM.elements[e]
            parameters = element._parameters
            _global_element_map_dict[e] = parameters

        __DDD__ = COMM.gather(_global_element_map_dict, root=MASTER_RANK)
        if RANK == MASTER_RANK:
            _global_element_map_dict = {}
            for _D_ in __DDD__:
                _global_element_map_dict.update(_D_)
            _BASE_['element_parameter_dict'] = _global_element_map_dict
        else:
            pass

        base['_BASE_'] = _BASE_
    else:
        pass

    COMM.barrier()

    return _BASE_


def ___link_all_forms____(the_great_mesh, base):
    r""""""
    partial_meshes = base['meshes']  # all the partial meshes
    spaces = base['spaces']          # all the msehtt spaces
    manifolds = base['manifolds']    # all the msehtt manifolds
    forms = base['forms']            # all the forms

    for f_sym in forms:
        form = forms[f_sym]
        abstract_form = form.abstract
        abstract_space = abstract_form.space._sym_repr
        abstract_mesh = abstract_form.mesh._sym_repr
        abstract_manifold = abstract_form.manifold._sym_repr

        form.current._tgm = the_great_mesh
        form.current._tpm = partial_meshes[abstract_mesh].current
        form.current._manifold = manifolds[abstract_manifold]
        form.current._space = spaces[abstract_space].current

    COMM.barrier()


def ___renew_stamp___(renew_stamp, trf, ts, base):
    r""""""
    base['stamp'] = renew_stamp

    for pur_lin_repr in base['forms']:
        form = base['forms'][pur_lin_repr]
        if RANK == MASTER_RANK:
            form.___renew_info___ = {
                'stamp': renew_stamp,
                'trf': trf,
                'ts': ts,
            }
        else:
            form.___renew_info___ = {
                'stamp': renew_stamp,
            }

        if form.___msehtt_base___ is None:
            form.___msehtt_base___ = base
        else:
            pass

    ___config_meshes___ = base['___config_meshes___']
    for mesh_sym_repr in ___config_meshes___:
        # to make sure meshes are initialized in the sequence as they configured.
        assert mesh_sym_repr in base['meshes']
        mesh = base['meshes'][mesh_sym_repr]
        mesh.___renew_stamp___ = renew_stamp

    for ab_sp_sym_repr in base['spaces']:
        space = base['spaces'][ab_sp_sym_repr]
        ab_sp = space._abstract_space

        if ab_sp.orientation == 'unknown':
            # These spaces are probably not for root-forms, skipping is OK.
            pass
        else:
            space.___renew_stamp___ = renew_stamp

    COMM.barrier()


def ___check_tgm___(TGM):
    r""""""
    elements = TGM.elements
    for e in elements:
        assert isinstance(e, int), f"BASE mesh element indices must be int."
