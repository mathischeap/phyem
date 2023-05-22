# -*- coding: utf-8 -*-
"""

@author: Yi Zhang.
         Department of Aerodynamics
         Faculty of Aerospace Engineering
         TU Delft, Delft, the Netherlands

"""
from abc import ABC
from time import localtime, strftime


class MyTimer(ABC):
    """My timers."""
    @classmethod
    def current_time(cls):
        """(str) Return a string showing current time."""
        return strftime("[%Y-%m-%d %H:%M:%S]", localtime())

    @classmethod
    def current_time_with_no_special_characters(cls):
        ct = cls.current_time()
        ct = ct.replace(' ', '_')
        ct = ct.replace('[', '')
        ct = ct.replace(']', '')
        ct = ct.replace(':', '_')
        ct = ct.replace('-', '_')
        return ct

    @classmethod
    def seconds2hms(cls, seconds):
        """We convert float: seconds to str: '[hh:mm:ss]'."""
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return '[%02d:%02d:%02d]' % (hours, minutes, seconds)

    @classmethod
    def seconds2hmsm(cls, seconds):
        """We convert float: seconds to str: '[hh:mm:ss.ms]'."""
        ms = (seconds - int(seconds)) * 1000
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return '[%02d:%02d:%02d.%03d]' % (hours, minutes, seconds, ms)

    @classmethod
    def seconds2dhms(cls, seconds):
        """We convert float: seconds to str: '[dd:hh:mm:ss]'."""
        days = int(seconds / 86400)
        SECONDS = seconds - 86400 * days
        minutes, SECONDS = divmod(SECONDS, 60)
        hours, minutes = divmod(minutes, 60)
        return '[{}:'.format(days) + '%02d:%02d:%02d]' % (hours, minutes, SECONDS)

    @classmethod
    def seconds2dhmsm(cls, seconds):
        """We convert float: seconds to str: '[dd:hh:mm:ss]'."""
        ms = (seconds - int(seconds)) * 1000
        days = int(seconds / 86400)
        SECONDS = seconds - 86400 * days
        minutes, SECONDS = divmod(SECONDS, 60)
        hours, minutes = divmod(minutes, 60)
        return '[{}:'.format(days) + '%02d:%02d:%02d.%03d]' % (hours, minutes, SECONDS, ms)

    @classmethod
    def hms2seconds(cls, hms):
        """We convert str: '[hh:mm:ss]' into float: seconds."""
        hh, mm, ss = hms[1:-1].split(':')
        hh = int(hh) * 3600
        mm = int(mm) * 60
        ss = int(ss)
        return hh + mm + ss

    @classmethod
    def dhms2seconds(cls, hms):
        """We convert str: '[hh:mm:ss]' into float: seconds."""
        dd, hh, mm, ss = hms[1:-1].split(':')
        dd = int(dd) * 86400
        hh = int(hh) * 3600
        mm = int(mm) * 60
        ss = int(ss)
        return dd + hh + mm + ss

    @classmethod
    def dhmsm2seconds(cls, hmsm):
        """We convert str: '[hh:mm:ss]' into float: seconds."""
        hms, ms = hmsm.split('.')
        ms = int(ms[:-1]) / 1000
        dd, hh, mm, ss = hms[1:].split(':')
        dd = int(dd) * 86400
        hh = int(hh) * 3600
        mm = int(mm) * 60
        ss = int(ss)
        return dd + hh + mm + ss + ms
