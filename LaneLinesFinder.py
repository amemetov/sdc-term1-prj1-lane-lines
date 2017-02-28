import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from collections import deque

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines1(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    if (lines is None):
        return

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, roi_apex):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


class LaneLines(object):
    def __init__(self, left_model, right_model):
        self.left_model = left_model
        self.right_model = right_model


class LaneLinesFinder(object):
    # Define default parameters for Canny
    CANNY_LOW_THRESHOLD = 100
    CANNY_HIGH_THRESHOLD = 170

    # Define default parameters for HoughTransform
    HT_RHO = 1  # distance resolution in pixels of the Hough grid
    HT_THETA = 1 * np.pi / 180  # angular resolution in radians of the Hough grid
    HT_THRESHOLD = 5  # minimum number of votes (intersections in Hough grid cell)
    HT_MIN_LINE_LEN = 5  # minimum number of pixels making up a line
    HT_MAX_LINE_GAP = 10  # maximum gap in pixels between connectable line segments

    def __init__(self, config=None,
                 line_render_thickness=10,
                 left_line_color=[255, 0, 0], right_line_color=[255, 0, 0],
                 avg_models_len=10, avg_models_weights_start=0.01, avg_models_weights_stop=0.1,
                 draw_origin_lines=False, store_images=False):
        self.result_dir = "video_result/"
        self.image_frame_idx = 0
        self.store_images = store_images
        self.draw_origin_lines = draw_origin_lines

        self.line_render_thickness = line_render_thickness
        self.left_line_color = left_line_color
        self.right_line_color = right_line_color

        self.extrapolate_consider_line_len = False
        self.extrapolate_step_len = 10

        self.avg_models_len = avg_models_len
        self.avg_models_weights_start = avg_models_weights_start
        self.avg_models_weights_stop = avg_models_weights_stop

        self.last_lane_models = deque([], maxlen=self.avg_models_len)

        if config is None: config = {}
        config.setdefault('canny_low_threshold', LaneLinesFinder.CANNY_LOW_THRESHOLD)
        config.setdefault('canny_high_threshold', LaneLinesFinder.CANNY_HIGH_THRESHOLD)
        config.setdefault('ht_rho', LaneLinesFinder.HT_RHO)
        config.setdefault('ht_theta', LaneLinesFinder.HT_THETA)
        config.setdefault('ht_threshold', LaneLinesFinder.HT_THRESHOLD)
        config.setdefault('ht_min_line_len', LaneLinesFinder.HT_MIN_LINE_LEN)
        config.setdefault('ht_max_line_gap', LaneLinesFinder.HT_MAX_LINE_GAP)

        self.config = config

    def process_image(self, image):
        # NOTE: The output you return should be a color image (3 channel) for processing video below
        # TODO: put your pipeline here,
        # you should return the final output (image where lines are drawn on lanes)

        (edges, roi_img, line_img, lines_edges) = self.find_lane_lines(image,
                                                                       self.config['canny_low_threshold'], self.config['canny_high_threshold'],
                                                                       self.config['ht_rho'], self.config['ht_theta'], self.config['ht_threshold'],
                                                                       self.config['ht_min_line_len'], self.config['ht_max_line_gap'])
        if self.store_images:
            mpimg.imsave(self.result_dir + "src_img-{0}.png".format(self.image_frame_idx), image)
            mpimg.imsave(self.result_dir + "edges-{0}.png".format(self.image_frame_idx), edges)
            mpimg.imsave(self.result_dir + "roi_img-{0}.png".format(self.image_frame_idx), roi_img)
            mpimg.imsave(self.result_dir + "line_img-{0}.png".format(self.image_frame_idx), line_img)
            mpimg.imsave(self.result_dir + "lines_edges-{0}.png".format(self.image_frame_idx), lines_edges)

            self.image_frame_idx += 1

        result = lines_edges
        return result

    def find_lane_lines(self, origin_img, canny_low_threshold, canny_high_threshold,
                        ht_rho, ht_theta, ht_threshold, ht_min_line_len, ht_max_line_gap):
        width = origin_img.shape[1]
        height = origin_img.shape[0]

        # Step 1. filtering white and yellow pixels
        filtered_image = self.filter_white_and_yellow_pixels(origin_img)

        # Step 2. grayscaling
        gray = grayscale(filtered_image)

        # Step 3. blurring using gaussian
        kernel_size = 3
        blur_gray = gaussian_blur(gray, kernel_size)

        # Step 4. finding edges using canny algo
        edges = canny(blur_gray, canny_low_threshold, canny_high_threshold)

        # Step 5. masking with ROI
        ysize = height
        xsize = width
        xdelta_bottom = 5
        xdelta_top = 0.05 * width
        xmiddle = width / 2
        roi_apex = (0.6) * height
        roi_vertices = np.array([[(xdelta_bottom, ysize), (xmiddle - xdelta_top, roi_apex),
                                  (xmiddle + xdelta_top, roi_apex), (xsize - xdelta_bottom, ysize)]], dtype=np.int32)

        masked_edges = region_of_interest(edges, roi_vertices)

        # create ROI image for further debugging
        roi_img = np.copy(origin_img)
        cv2.polylines(roi_img, roi_vertices, isClosed=False, color=(0, 0, 255), thickness=2)

        # Step 6. finding lines using HoughTransform and drawing found lines
        lines = hough_lines(masked_edges, ht_rho, ht_theta, ht_threshold, ht_min_line_len, ht_max_line_gap, roi_apex)
        (line_img, left_model, right_model) = self.draw_lines(width, height, lines, roi_apex)

        # store models for further using
        self.last_lane_models.append(LaneLines(left_model, right_model))

        # Step 7. Draw the lines on the origin image
        lines_edges = weighted_img(line_img, origin_img, α=0.7, β=1., λ=0.)

        return (edges, roi_img, line_img, lines_edges)

    def draw_lines(self, width, height, lines, roi_apex):
        img = np.zeros((height, width, 3), dtype=np.uint8)

        if (lines is None):
            return img

        if self.draw_origin_lines:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(img, (x1, y1), (x2, y2), [255, 0, 0], 2)
            # return

        height = img.shape[0]
        width = img.shape[1]

        sorted_lines = self.sort_lines_by_y(lines)
        left_lines, right_lines = self.split_lines(sorted_lines, width)

        (min_y, max_y) = (roi_apex, height)

        left_model = self.fit_lines_and_render(img, left_lines, min_y, max_y, self.left_line_color, self.line_render_thickness, True)
        right_model = self.fit_lines_and_render(img, right_lines, min_y, max_y, self.right_line_color, self.line_render_thickness, False)

        return (img, left_model, right_model)

    def fit_lines_and_render(self, img, lines, min_y, max_y, color, thickness, is_left):
        if len(lines) > 0:
            left_lines_x, left_lines_y = self.avg_line(lines)

            # we need to find x by y value, so change order of arguments
            result_model = np.poly1d(np.polyfit(left_lines_y, left_lines_x, 1))
            result_model = self.avg_model(result_model, is_left)

            left_x_start = result_model(max_y)
            left_x_end = result_model(min_y)

            cv2.line(img, (int(left_x_start), int(max_y)), (int(left_x_end), int(min_y)), color, thickness)

            return result_model

        return None

    def avg_model(self, model, is_left):
        models_num = len(self.last_lane_models)
        if models_num == 0:
            return model

        weights = np.linspace(self.avg_models_weights_start, self.avg_models_weights_stop, num=models_num)
        idx = 0
        coeffs_accum = None
        for lane_model in self.last_lane_models:
            m = lane_model.left_model if is_left else lane_model.right_model
            coeffs = np.copy(m.coeffs) if m is not None else np.zeros(2)
            coeffs *= weights[idx]
            if coeffs_accum is None:
                coeffs_accum = coeffs
            else:
                coeffs_accum += coeffs
            idx += 1

        coeffs_accum += model.coeffs
        result_coeffs = coeffs_accum / (np.sum(weights) + 1)

        return np.poly1d(result_coeffs)


    def avg_line(self, lines):
        lines_x = []
        lines_y = []
        for l in lines:
            len = l.x2 - l.x1
            len_abs = math.fabs(len)

            lines_x.append(int(l.x1))
            lines_y.append(int(l.y1))

            # add more points for longer lines to avoid influence of noisy short lines
            if self.extrapolate_consider_line_len:
                if len_abs >= 2 * self.extrapolate_step_len:
                    step_num = int(len / self.extrapolate_step_len)
                    # poly = np.poly1d(np.polyfit([l.x1, l.x2], [l.y1, l.y2], 1))
                    for s in range(1, step_num):
                        x = l.x1 + s * self.extrapolate_step_len
                        # y = poly(x)
                        y = l.calc_y(x)
                        lines_x.append(int(x))
                        lines_y.append(int(y))

            lines_x.append(int(l.x2))
            lines_y.append(int(l.y2))
        return (lines_x, lines_y)

    def split_lines(self, lines, width):
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

    def filter_white_and_yellow_pixels(self, origin_img):
        hsv = cv2.cvtColor(origin_img, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 200], dtype=np.uint8)
        upper_white = np.array([180, 255, 255], dtype=np.uint8)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        res_white = cv2.bitwise_and(origin_img, origin_img, mask=mask_white)

        # lower_yellow = np.array([90, 100, 100])
        # upper_yellow = np.array([110, 255, 255])
        # mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # res_yellow = cv2.bitwise_and(origin_img, origin_img, mask=mask_yellow)
        #
        # return cv2.addWeighted(res_white, 1.0, res_yellow, 1., 0.)
        # return cv2.bitwise_or(res_white, res_yellow)

        return res_white

    def sort_lines_by_y(self, lines):
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
