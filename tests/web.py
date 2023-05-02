# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang

Regenerate the web page.
"""

import os
import sys

if './' not in sys.path:
    sys.path.append('./')

stream = os.popen(rf'sphinx-build -b html web\source web\build\html')
output = stream.read()
print(output)


if __name__ == '__main__':
    # python tests/web.py
    pass
