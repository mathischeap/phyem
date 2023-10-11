r"""

"""
from src.config import RANK
from generic.py._2d_unstruct.space.main import GenericUnstructuredSpace2D

from _MPI.generic.py._2d_unstruct.space.gathering_matrix.main import MPI_PY2_GatheringMatrix
from _MPI.generic.py._2d_unstruct.space.mass_matrix.main import MPI_PY2_MassMatrix
from _MPI.generic.py._2d_unstruct.space.incidence_matrix.main import MPI_PY2_IncidenceMatrix


class MPI_Py_2D_Unstructured_Space(GenericUnstructuredSpace2D):
    """"""

    def __init__(self, *args):
        super().__init__(*args)
        self._gm = MPI_PY2_GatheringMatrix(self)
        self._E = MPI_PY2_IncidenceMatrix(self)
        self._M = MPI_PY2_MassMatrix(self)

    def __repr__(self):
        """Repr"""
        return f"<Py2-Space-RANK{RANK} of {self.abstract} at {self._signature}>"
