# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 7:00 PM on 5/2/2023
"""

import os
import sys
if './' not in sys.path:
    sys.path.append('./')

__all__ = [
    'msepy',
]

from src.config import SIZE


if SIZE == 1:

    print(  # we use a container to do the test for safety reasons.
        os.popen('python tests/unittests/msepy/main.py').read()
    )

else:

    pass


if __name__ == '__main__':
    # python .\tests\unittests\main.py
    pass
