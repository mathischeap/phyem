r"""Wrapper of methods for pass something one to another.

"""

from phyem.tools.frozen import Frozen


class MseHtt_MultiGrid_GreatMesh_Pass(Frozen):
    r""""""

    def __init__(self, tgm):
        r""""""
        self._tgm = tgm
        self._freeze()

    @staticmethod
    def cochain(ff, from_time, tf, to_time):
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

        """
        tsp = ff.numeric.tsp(from_time)
        tf[to_time].reduce(tsp[from_time])
