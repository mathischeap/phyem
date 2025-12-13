r"""
python tests/tools/geometries_m2n2.py
"""

from phyem.tools.miscellaneous.geometries.m2n2 import *

p0 = Point2(0, 0)
p1 = Point2(1, 0)
p2 = Point2(1, 1)
p3 = Point2(0, 1)

sg0 = StraightSegment2(p0, p1)
sg1 = StraightSegment2(p1, p2)
sg2 = StraightSegment2(p2, p3)
sg3 = StraightSegment2(p3, p0)

sg = find_intersection_of_two_parallel_straight_segments(sg0, sg2)
assert sg is None
assert whether_point_on_straight_segment(p0, sg0)
assert whether_point_on_straight_segment(p1, sg0)

p4 = Point2(0.5, 0)
assert whether_point_on_straight_segment(p4, sg0)
assert not whether_point_on_straight_segment(p4, sg1)
p5 = Point2(1, 0.25)
assert whether_point_on_straight_segment(p5, sg1)

p6 = Point2(1, 0.5)
p7 = Point2(1.5, 1)

line = StraightLine2(p4, p6)
assert whether_point_on_straight_line(p7, line)
line = StraightLine2(p4, p7)
assert whether_point_on_straight_line(p6, line)
assert distance2(p4, p6) == distance2(p6, p7)
assert whether_two_lines_or_segments_parallel(sg0, sg2)
assert whether_two_lines_or_segments_parallel(sg1, sg3)
assert not whether_two_lines_or_segments_parallel(sg1, sg0)
assert line.slope == 1
p8 = Point2(0, 0.5)
sg5 = StraightSegment2(p8, p4)
assert sg5.slope == -1

p9 = Point2(2, 0)
sg6 = StraightSegment2(p0, p9)
sg = find_intersection_of_two_parallel_straight_segments(sg0, sg6)
assert sg == sg0
sg7 = StraightSegment2(p4, Point2(-0.5, 0))
sg = find_intersection_of_two_parallel_straight_segments(sg7, sg6)
assert sg == StraightSegment2(p0, p4)
assert StraightSegment2(p0, p4) == StraightSegment2(p4, p0)

poly0 = Polygon2(p0, p1, p2, p3)

p10 = Point2(2, 0.25)
p11 = Point2(2, 1)
poly1 = Polygon2(p5, p10, p11, p2)

assert poly0.area == 1 and poly1.area == 0.75
poly2 = Polygon2(p5, p10, p11, p3)

union0 = poly0.union(poly1)
union1 = poly0.union(poly2)
assert union0.area == union1.area == 1.75

line = poly0.intersection(poly1)

poly3 = Polygon2(p0, p1, p5, p3)
point = poly3.intersection(poly1)
assert point == p5
