# -*- coding: utf-8 -*-
r"""
"""
__all__ = [
    'wedge',
    'Hodge',
    'd', 'exterior_derivative', 'codifferential',
    'inner',
    'time_derivative',
    'trace',
]

from src.form.operators import wedge

from src.form.operators import Hodge

from src.form.operators import d
exterior_derivative = d   # `exterior_derivative` is equivalent to `d`.

from src.form.operators import codifferential

from src.wf.term.main import inner

from src.form.operators import time_derivative

from src.form.operators import trace
