# -*- coding: utf-8 -*-
r"""
"""
__all__ = [
    'Point2',
    'StraightLine2',
    'distance2',
    'line2_line2_intersection',   # to find the intersection (if exists) of two straight lines (not segments).
    'Polygon2',
    'Curve2',
    'StraightSegment2',
    'whether_point_on_straight_line',     # m2n2
    'whether_point_on_curve',             # m2n2
    'whether_point_in_polygon',           # m2n2
    'whether_point_on_straight_segment',  # m2n2

    'Point3',                             # m3n3
    'PerpRectangle',                      # m3n3
    'OrthogonalSegment3',                 # m3n3
    'OrthogonalHexahedron',               # m3n3
]

from phyem.tools.miscellaneous.geometries.m2n2 import Point2
from phyem.tools.miscellaneous.geometries.m2n2 import StraightLine2
from phyem.tools.miscellaneous.geometries.m2n2 import distance2
from phyem.tools.miscellaneous.geometries.m2n2 import line2_line2_intersection
from phyem.tools.miscellaneous.geometries.m2n2 import Polygon2
from phyem.tools.miscellaneous.geometries.m2n2 import Curve2
from phyem.tools.miscellaneous.geometries.m2n2 import StraightSegment2
from phyem.tools.miscellaneous.geometries.m2n2 import whether_point_on_straight_line
from phyem.tools.miscellaneous.geometries.m2n2 import whether_point_on_curve
from phyem.tools.miscellaneous.geometries.m2n2 import whether_point_in_polygon
from phyem.tools.miscellaneous.geometries.m2n2 import whether_point_on_straight_segment

from phyem.tools.miscellaneous.geometries.m3n3 import Point3
from phyem.tools.miscellaneous.geometries.m3n3 import PerpRectangle
from phyem.tools.miscellaneous.geometries.m3n3 import OrthogonalSegment3
from phyem.tools.miscellaneous.geometries.m3n3 import OrthogonalHexahedron
