import numpy as np
import cv2
import math

def draw_lines(img, lines, roi_apex, color=[255, 0, 0], thickness=6):
    if (lines is None):
        return

    # draw origin lines
    if False:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        #return


    height = img.shape[0]
    width = img.shape[1]

    sorted_lines = sort_lines_by_y(lines)
    left_lines, right_lines = split_lines(sorted_lines, width, height)

    # debug split lines
    if False:
        for l in left_lines:
            cv2.line(img, (l.x1, l.y1), (l.x2, l.y2), [0, 255, 0], thickness)

        for l in right_lines:
            cv2.line(img, (l.x1, l.y1), (l.x2, l.y2), [0, 0, 255], thickness)

        return

    (min_y, max_y) = (roi_apex, height)#calc_y_interval(left_lines, right_lines, height, roi_apex)

    if len(left_lines) > 0:
        left_lines_x, left_lines_y = avg_line(left_lines, gen_inner_points=True)

        # we need to find x by y value, so change order of arguments
        left_reg = np.poly1d(np.polyfit(left_lines_y, left_lines_x, 1))
        left_x_start = left_reg(max_y)
        left_x_end = left_reg(min_y)

        cv2.line(img, (int(left_x_start), int(max_y)), (int(left_x_end), int(min_y)), [0, 255, 0], thickness)

    if len(right_lines) > 0:
        right_lines_x, right_lines_y = avg_line(right_lines, gen_inner_points=True)

        right_reg = np.poly1d(np.polyfit(right_lines_y, right_lines_x, 1))
        right_x_start = right_reg(max_y)
        right_x_end = right_reg(min_y)

        cv2.line(img, (int(right_x_start), int(max_y)), (int(right_x_end), int(min_y)), [0, 0, 255], thickness)

def avg_line(lines, gen_inner_points=True):
    # add more points for longer lines to avoid effect of noisy short lines

    step_len = 10

    lines_x = []
    lines_y = []
    for l in lines:
        len = l.x2 - l.x1
        len_abs = math.fabs(len)

        lines_x.append(int(l.x1))
        lines_y.append(int(l.y1))

        if gen_inner_points:
            if len_abs >= 2*step_len:
                step_num = int(len / step_len)
                #poly = np.poly1d(np.polyfit([l.x1, l.x2], [l.y1, l.y2], 1))
                for s in range(1, step_num):
                    x = l.x1 + s * step_len
                    #y = poly(x)
                    y = l.calc_y(x)
                    lines_x.append(int(x))
                    lines_y.append(int(y))

        lines_x.append(int(l.x2))
        lines_y.append(int(l.y2))
    return (lines_x, lines_y)


def calc_y_interval(left_lines, right_lines, height, roi_apex):
    max_y = height - 10
    min_y = roi_apex#height / 2

    max_y = max(left_lines[0].y1, max_y)
    min_y = min(left_lines[-1].y2, min_y)

    max_y = max(right_lines[0].y1, max_y)
    min_y = min(right_lines[-1].y2, min_y)

    return (min_y, max_y)


def split_lines(lines, width, height):
    middle_x = width / 2

    ox = Line(0, 0, 1, 0)

    left_lines_angle_threshold_start = math.radians(-30)
    left_lines_angle_threshold_end = math.radians(-60)

    right_lines_angle_threshold_start = math.radians(-120)
    right_lines_angle_threshold_end = math.radians(-155)

    left_lines = []
    right_lines = []

    for l in lines:
        angle = ox.angle_between_lines2(l)

        if l.x1 < middle_x and left_lines_angle_threshold_end <= angle <= left_lines_angle_threshold_start:
            left_lines.append(l)
        elif l.x1 > middle_x and right_lines_angle_threshold_end <= angle <= right_lines_angle_threshold_start:
            right_lines.append(l)

    return (left_lines, right_lines)


def split_lines_old(lines, width, height):
    # lines is list of Line instances sorted by y1 in reverse order

    middle_x = width / 2

    #max_dist = 10
    max_straight_angle = math.radians(5)
    #max_rotate_angle = math.radians(30)

    max_x_diff_on_top = 25 # px

    #ox = Line(0, 0, 1, 0)

    left_lines_candidate_groups = [] #array of array
    right_lines_candidate_groups = [] #array of array

    for line in lines:
        slope = line.slope()

        x_top = line.calc_x(height)

        if slope < 0 and line.x1 < middle_x and 0 <= x_top < middle_x:
            for left_lines in left_lines_candidate_groups:
                left_line = left_lines[0]
                left_line_x_top = left_line.calc_x(height)

                angle = left_line.angle_between_lines2(line)

                if left_line_x_top - max_x_diff_on_top <= x_top <= left_line_x_top + max_x_diff_on_top \
                        and math.fabs(angle) < max_straight_angle:
                    left_lines.append(line)
                    continue

            # not found group - create new group
            left_lines_candidate_groups.append([line])
        elif slope >= 0 and line.x1 > middle_x and middle_x < x_top <= width:
            for right_lines in right_lines_candidate_groups:
                right_line = right_lines[0]
                right_line_x_top = right_line.calc_x(height)

                angle = right_line.angle_between_lines2(line)

                if right_line_x_top - max_x_diff_on_top <= x_top <= right_line_x_top + max_x_diff_on_top \
                        and math.fabs(angle) < max_straight_angle:
                    right_lines.append(line)
                    continue

            # not found group - create new group
            right_lines_candidate_groups.append([line])

    return (get_result(left_lines_candidate_groups), get_result(right_lines_candidate_groups))


def get_result(lines_candidate_groups):
    result_lines = None
    max_num = -1
    for lines in lines_candidate_groups:
        total_len = 0
        for l in lines:
            total_len += l.lengthSq()

        #num = len(lines)
        num = total_len
        if num > max_num:
            max_num = num
            result_lines = lines
    return result_lines


def sort_lines_by_y(lines):
    # sort by Y values
    sorted_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            l = Line(x1, y1, x2, y2) if y1 > y2 else Line(x2, y2, x1, y1)
            if l.is_horizontal():
                continue
            sorted_lines.append(l)

    sorted_lines.sort(key=lambda l: l.y1, reverse=True)
    return sorted_lines



class Line(object):
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

        # y = mx + b
        # Ax + By + C = 0
        self.a = y1 - y2
        self.b = x2 - x1
        self.c = x1*y2 - x2*y1

    def slope(self):
        return (self.y2 - self.y1) / (self.x2 - self.x1)

    def calc_x(self, y):
        return (-self.b*y - self.c)/self.a

    def calc_y(self, x):
        return (-self.a*x - self.c)/self.b

    def length(self):
        return math.sqrt(self.lengthSq())

    def lengthSq(self):
        ax = self.x2 - self.x1
        ay = self.y2 - self.y1
        return (ax*ax + ay*ay)

    def angle_between_lines(self, line):
        ax = self.x2 - self.x1
        ay = self.y2 - self.y1
        bx = line.x2 - line.x1
        by = line.y2 - line.y1
        dot_product = ax*bx + ay*by
        cos = dot_product/(self.length()*line.length())
        cos = max(-1, cos)
        cos = min(1, cos)

        return math.acos(cos)

    def angle_between_lines2(self, line):
        ax = self.x2 - self.x1
        ay = self.y2 - self.y1
        bx = line.x2 - line.x1
        by = line.y2 - line.y1
        return math.atan2(by, bx) - math.atan2(ay, ax)

    def distance_to_pt(self, pt_x, pt_y):
        return math.fabs(self.a * pt_x + self.b * pt_y + self.c) / math.sqrt(self.a * self.a + self.b * self.b)

    def is_almost_equal(self, line, eps=0.1):
        return math.fabs(self.a - line.a) < eps and math.fabs(self.b - line.b) < eps and math.fabs(self.c - line.c) < eps

    def is_almost_parallel(self, line, eps=0.1):
        if self.a == 0 or line.a == 0:
            # both horizontal?
            return self.a == line.a

        if self.b == 0 or line.b == 0:
            # both vertical ?
            return self.b == line.b

        return math.fabs(self.a / line.a - self.b / line.b) < eps

    def is_horizontal(self, eps=0.1):
        return math.fabs(self.a) < eps

    def is_vertical(self):
        return self.b == 0

    def __str__(self):
        return "({0},{1}) -> ({2},{3})".format(self.x1, self.y1, self.x2, self.y2)

    def __repr__(self):
        return str(self)





