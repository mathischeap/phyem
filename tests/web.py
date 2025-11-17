# -*- coding: utf-8 -*-
r"""
To build the web, do

$ sphinx-build -b html web\source web\build\html

To run this module, do

$ python tests/web.py

This will:
    1) Do all doctests.
    2) Regenerate the web page.

"""

import os

if os.path.isfile(f"./web/source/conf.py"):   # do all doctests in we have the conf.py file.
    stream = os.popen(rf'.\web\make doctest')
    output = stream.read()
    print(output)

if os.path.isfile(f"./web/source/conf.py"):   # regenerate the web if we have the conf.py file.
    stream = os.popen(rf'sphinx-build -b html web\source web\build\html')
    output = stream.read()
    print(output)
