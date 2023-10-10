r"""

"""
from src.config import RANK
from generic.py._2d_unstruct.space.main import GenericUnstructuredSpace2D


class MPI_Py_2D_Unstructured_Space(GenericUnstructuredSpace2D):
    """"""

    def __repr__(self):
        """Repr"""
        return f"<Py2-Space-RANK{RANK} of {self.abstract} at {self._signature}>"
