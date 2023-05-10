# -*- coding: utf-8 -*-
r"""
@author: Yi Zhang
@contact: zhangyi_aero@hotmail.com
@time: 11/26/2022 2:56 PM

To build the web:
$ sphinx-build -b html web\source web\build\html

"""

import os
absolute_path = os.path.dirname(__file__)
import sys
if absolute_path not in sys.path:
    sys.path.append(absolute_path)

__version__ = '0.0.0'

__all__ = [
    'config',
    'list_forms', 'list_spaces', 'list_meshes',
    'clear_forms',
    'samples',
    'manifold',
    'mesh',
    'space',
    'inner', 'wedge', 'Hodge',
    'd', 'exterior_derivative', 'trace',
    'codifferential',
    'time_derivative',
    'pde',
    'ode',
    'time_sequence',
    'constant_scalar',

    'fem',
    'vc',
]

import src.config as config

from src.form.others import _list_forms as list_forms
from src.spaces.main import _list_spaces as list_spaces
from src.mesh import _list_meshes as list_meshes

from src.form.main import _clear_forms as clear_forms

import tests.samples.main as samples

from src.manifold import manifold

from src.mesh import mesh

import src.spaces.main as space

from src.operators import inner, wedge, Hodge, trace
from src.operators import d, exterior_derivative
from src.operators import codifferential
from src.operators import time_derivative

from src.pde import pde
from src.ode.main import ode

from src.tools.time_sequence import abstract_time_sequence as time_sequence
from src.form.parameters import constant_scalar

import src.fem as fem

import tools.vector_calculus as vc
