import numpy as np
import cv2
import math

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
            total_len += l.length_sq()

        #num = len(lines)
        num = total_len
        if num > max_num:
            max_num = num
            result_lines = lines
    return result_lines

def calc_y_interval(left_lines, right_lines, height, roi_apex):
    max_y = height - 10
    min_y = roi_apex#height / 2

    max_y = max(left_lines[0].y1, max_y)
    min_y = min(left_lines[-1].y2, min_y)

    max_y = max(right_lines[0].y1, max_y)
    min_y = min(right_lines[-1].y2, min_y)

    return (min_y, max_y)


