# -*- coding: utf-8 -*-
r"""
"""
__all__ = [
    'wedge',
    'Hodge',
    'd', 'exterior_derivative', 'codifferential',
    'inner',
    'dp',
    'time_derivative',
    'trace',
]

from phyem.src.form.operators import wedge

from phyem.src.form.operators import Hodge

from phyem.src.form.operators import d
exterior_derivative = d   # `exterior_derivative` is equivalent to `d`.

from phyem.src.form.operators import codifferential

from phyem.src.wf.term.main import inner
from phyem.src.wf.term.main import duality_pairing as dp


from phyem.src.form.operators import time_derivative

from phyem.src.form.operators import trace
