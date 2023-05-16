# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 7:00 PM on 5/2/2023
"""
import sys
if './' not in sys.path:
    sys.path.append('./')

__all__ = [
    'msepy',
]

from src.config import SIZE

unittest_error_report = list()

if SIZE == 1:

    import tests.unittests.msepy.main as msepy

    unittest_error_report.extend(
        msepy.msepy_error_report
    )

else:

    pass


if __name__ == '__main__':
    # python .\tests\unittests\main.py
    pass
