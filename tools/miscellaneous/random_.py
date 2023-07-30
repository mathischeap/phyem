# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 5:41 PM on 7/30/2023
"""
import random
import string


def string_digits(stringLength=8):
    """Generate a random string of letters and digits."""
    lettersAndDigits = string.ascii_letters + string.digits
    return ''.join(random.choice(lettersAndDigits) for _ in range(stringLength))
