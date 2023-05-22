""""""

from tools.frozen import Frozen


class Iterator(Frozen):
    """Iterator over time steps."""

    def __init__(self, dynamic_linear_system):
        """Initialize an iterator with a dynamic linear system."""
        self._dls = dynamic_linear_system
        self._freeze()


