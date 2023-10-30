# -*- coding: utf-8 -*-
r"""
"""
import random
import string


def string_digits(stringLength=8):
    """Generate a random string of letters and digits."""
    lettersAndDigits = string.ascii_letters + string.digits
    return ''.join(random.choice(lettersAndDigits) for _ in range(stringLength))
