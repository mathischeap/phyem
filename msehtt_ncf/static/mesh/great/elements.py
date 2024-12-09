# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from src.config import RANK, MASTER_RANK, COMM


class MseHtt_NCF_GreatMesh_Elements(Frozen):
    r""""""

    def __init__(self, tgm, elements_dict):
        r""""""
        self._tgm = tgm
        self._elements_dict = elements_dict
        self._mn = None
        self._periodic_face_pairing = None
        self._parse_statistics()
        self._visualize = None
        self._ncf_pairing_ = self._parse_ncf_pairing_()
        self._freeze()

    def __repr__(self):
        r""""""
        return f"<elements of {self._tgm}>"

    @property
    def ___is_msehtt_ncf_great_mesh_elements___(self):
        r"""Just a signature."""
        return True

    @property
    def tgm(self):
        r"""Return the great mesh I am built on."""
        return self._tgm

    def __getitem__(self, item):
        r"""Return the element indexed ``item``."""
        return self._elements_dict[item]

    def __len__(self):
        r"""How many elements in this rank?"""
        return len(self._elements_dict)

    def __contains__(self, item):
        r"""If the element indexed ``item`` is an valid element in this rank?"""
        return item in self._elements_dict

    def __iter__(self):
        r"""Iterate over all element indices in this rank."""
        for element_index in self._elements_dict:
            yield element_index

    def map_(self, i):
        r"""Return the element map of a local element."""
        return self[i].map_

    @property
    def global_map(self):
        r"""The element map of all elements across all ranks. This information is usually only stored in the
        master rank.
        """
        if RANK == MASTER_RANK:
            return self._tgm._global_element_map_dict
        else:
            return None

    @property
    def global_etype(self):
        r"""The element type of all elements across all ranks. This information is usually only stored in the
        master rank.
        """
        if RANK == MASTER_RANK:
            return self._tgm._global_element_type_dict
        else:
            return None

    @property
    def mn(self):
        r"""The `m` and `n` of the elements I have. All the elements across all ranks will take into account.
        So if the elements in a local rank are of one (m, n), (m, n) could be different in other ranks. So
        self.mn will return a tuple if two pairs of (m, n).

        For example, if self.mn = (2, 2), then all the elements I have are 2d (n=2) elements in 2d (m=2) space.

        If self.mn == ((3, 2), (2, 2)), then some of the elements I have are 2d (n=2) elements in 3d (m=2) space,
        and some other elements are 2d (n=2) element in 2d (m=2) space. Of course, this is not very likely.
        """
        if self._mn is None:
            mn_pool = list()
            for element_index in self:
                element = self[element_index]
                mn = (element.m(), element.n())
                if mn not in mn_pool:
                    mn_pool.append(mn)
                else:
                    pass
            mn_pool = COMM.gather(mn_pool, root=MASTER_RANK)
            if RANK == MASTER_RANK:
                total_mn_pool = set()
                for _ in mn_pool:
                    total_mn_pool.update(_)
                total_mn_pool = list(total_mn_pool)

                if len(total_mn_pool) == 1:
                    self._mn = total_mn_pool[0]
                else:
                    self._mn = tuple(total_mn_pool)
            else:
                self._mn = None

            self._mn = COMM.bcast(self._mn, root=MASTER_RANK)
        return self._mn

    @property
    def statistics(self):
        r"""Return some global statistic numbers. Since they are global, they will be same in all ranks."""
        return self._statistics

    def _parse_statistics(self):
        r"""Parse some global statistics. Return same in all ranks."""
        etype_pool = {}
        for i in self:
            element = self[i]
            etype = element.etype
            if etype not in etype_pool:
                etype_pool[etype] = 0
            else:
                pass
            etype_pool[etype] += 1

        rank_etype = COMM.gather(etype_pool, root=MASTER_RANK)

        if RANK == MASTER_RANK:

            etype_pool = {}
            for pool in rank_etype:
                for etype in pool:
                    if etype not in etype_pool:
                        etype_pool[etype] = 0
                    else:
                        pass
                    etype_pool[etype] += pool[etype]

            total_amount_elements = 0
            for etype in etype_pool:
                total_amount_elements += etype_pool[etype]

            statistics = {
                'total amount elements': total_amount_elements,
                'amount element types': len(etype_pool),
                'total amount different elements': etype_pool
            }

        else:
            statistics = None

        self._statistics = COMM.bcast(statistics, root=MASTER_RANK)

    def _parse_ncf_pairing_(self):
        r""""""
        ncf_pairing = {}
        global_map = self.global_map
        if RANK == MASTER_RANK:
            pass
        else:
            pass
        return ncf_pairing

    @property
    def periodic_face_pairing(self):
        r"""A dictionary shows the periodic pairing.

        For example,
            in 2d:

                periodic_face_pairing = {

                }

        """
        if self._periodic_face_pairing is not None:
            return self._periodic_face_pairing
        else:
            pass

        if self.mn == (2, 2):
            self._periodic_face_pairing = self.___periodic_face_pairing_m2n2___()
        else:
            raise NotImplementedError(f"not implemented for (m,n) == {self.mn}")

        return self._periodic_face_pairing

    def ___periodic_face_pairing_m2n2___(self):
        r""""""
        raise NotImplementedError()
