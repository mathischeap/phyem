# -*- coding: utf-8 -*-
r"""
pH-lib@RAM-EEMCS-UT
Yi Zhang

$ sphinx-build -b html web\source web\build\html

Regenerate the web page.
"""

import os
import sys

if './' not in sys.path:
    sys.path.append('./')

if os.path.isfile(f"./web/source/conf.py"):
    stream = os.popen(rf'.\web\make doctest')
    output = stream.read()
    print(output)

if os.path.isfile(f"./web/source/conf.py"):
    stream = os.popen(rf'sphinx-build -b html web\source web\build\html')
    output = stream.read()
    print(output)


if __name__ == '__main__':
    # python tests/web.py
    pass
