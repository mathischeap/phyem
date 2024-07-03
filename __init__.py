# -*- coding: utf-8 -*-
r"""
"""

import os
absolute_path = os.path.dirname(__file__)
import sys
if absolute_path not in sys.path:
    sys.path.append(absolute_path)

__version__ = '0.0.2'

__all__ = [
    'config',

    'tools',

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
    'wf',

    'time_sequence',
    'constant_scalar',

    'fem',
    'vc',

    "iterator",

    "run",  # runner
    'rdr',  # runner data reader

    "save",
    "read",

    "os",
    "film",

    "find_manifold",

    'vtk',  # save an object or some objects to a vtk file.
    'rws',  # save an object or some objects to a dds-rws-grouped file.

]

import src.config as config

from src.form.others import _list_forms as list_forms
from src.spaces.main import _list_spaces as list_spaces
from src.mesh import _list_meshes as list_meshes

from src.form.main import _clear_forms as clear_forms

import tests.samples.main as samples

from src.manifold import manifold
from src.manifold import find_manifold

from src.mesh import mesh

import src.spaces.main as space

from src.operators import inner, wedge, Hodge, trace
from src.operators import d, exterior_derivative
from src.operators import codifferential
from src.operators import time_derivative

from src.pde import pde
from src.ode.main import ode
from src.wf.main import WeakFormulation as wf

from src.time_sequence import abstract_time_sequence as time_sequence
from src.form.parameters import constant_scalar

import src.fem as fem

import tools.vector_calculus as vc

from tools.runner import ParallelMatrix3dInputRunner as run  # runner date reader
from tools.runner import RunnerDataReader as rdr             # runner date reader

from tools.iterator.main import Iterator as iterator

from tools.save import save
from tools.read import read

import tools.os_ as os
import tools.film as film

import tools

from tools.vtk_.main import vtk
from tools.dds.saving_api import _rws_grouped_saving as rws
