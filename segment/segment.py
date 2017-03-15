from matplotlib import pyplot as plt
from lib import dataset
from lib import imgio as ImgIO
from lib import process
from feature import process2
import numpy as np
import os
import random


class Separator(object):

    def __init__(self, image, output_shape=(70,70), foreground=None, length=-1, pixel_tolerance=1):
        self.image = image
        self.output_shape = output_shape
        self.objects = {}
        self.objects_boundaries = []
        self.char_images = []
        self.foreground = foreground if foreground else process2.get_background_color(image, foreground=True)
        self.background = process2.get_background_color(self.image, foreground=False)
        self.pixel_tolerance = pixel_tolerance


    def __call__(self, image, *args, **kwargs):
        self.image = image
        return self.segment_process(*args, **kwargs)

    def vertical_segment(self):
        height = self.image.shape[0]
        col_histogram = process2.pixels_histogram(self.image, axis=0, color=self.foreground)
        boundary_list = []
        start_flag = False
        for index in range(len(col_histogram)):
            if col_histogram[index] > self.pixel_tolerance:
                if col_histogram[min(index+1, len(col_histogram))] > self.pixel_tolerance and not start_flag:
                    boundary_list.append(index)
                    start_flag = True
                    continue
                elif col_histogram[min(index+1, len(col_histogram))] < self.pixel_tolerance and start_flag:
                    boundary_list.append(index)
                    start_flag = False
                    continue
        if not len(boundary_list) % 2  == 0:
            boundary_list.append(len(col_histogram))
        for index in range(0, len(boundary_list), 2):
            lower, upper = boundary_list[index], boundary_list[index+1]
            object_coordinates = [(i, j) for i in range(height) for j in range(lower, upper+1) if self.image[i, j] == self.foreground]
            ceiling, floor = self.get_ceiling_floor(object_coordinates)
            boundary = (ceiling-1, lower-1, floor+1, upper+1)
            self.objects_boundaries.append(boundary)
            self.objects[boundary] = object_coordinates

    def display_segment_result(self, boundaries=None, interpolation=None):
        if not boundaries:
            boundaries = self.objects_boundaries
        plt.figure(num='segment result', figsize=(8,8))
        #'astronaut')
        plt_image = self.image * 255.0 if np.max(self.image) <= 1.0 else self.image
        plt.imshow(np.uint8(plt_image), interpolation)
        if plt_image.ndim <= 2:
            plt.gray()
        for min_row, min_col, max_row, max_col in boundaries:
            x = [min_row, min_row, max_row, max_row, min_row]
            y = [min_col, max_col, max_col, min_col, min_col]
            plt.plot(y, x, 'r-', linewidth=2)
        plt.axis('equal')
        plt.axis('off')
        plt.show()

    def save_segment_result(self, folder, label, save_char=False):
        if not self.objects:
            self.get_characters()
        if save_char:
            if not os.path.exists(folder):
                os.makedirs(folder)
            for image, (index, l) in zip(self.char_images, enumerate(label)):
                char_folder = os.path.join(folder, l)
                if not os.path.exists(char_folder):
                    os.mkdir(char_folder)
                ImgIO.write_image(image, dataset.get_save_image_name(char_folder, str(index), img_type='jpg'))

        height, width = self.output_shape
        width *= len(self.objects)
        image = process.filter_scale(self.image, width=width, height=height)
        #initial = process2.max_min_normalization(process2.rgb_to_gray(self.image), max_value=1.0, min_value=0.0)
        #process2.resize_transform(initial, output_shape=(height, width))
        for img in self.char_images:
            image = np.hstack((image, np.zeros((height, 3)), img))
        image = np.hstack((image, np.zeros((height, 3))))
        filename = dataset.get_save_image_name(folder, label+'_segment_result', img_type='jpg')
        ImgIO.write_image(image, filename)

    def make_characters_images(self):
        offset = int(round(self.output_shape[0] * 0.15))
        def char_image_add_margin(img):
            img_height, img_width = img.shape
            if img_height >= self.output_shape[0] or img_width >= self.output_shape[1] :
                img = process2.normalise_scaling(img, output_shape=(self.output_shape[0]-offset, self.output_shape[1]-offset))
            new_image = np.ones(self.output_shape) * self.background
            h_gap = int(round((self.output_shape[0] - img.shape[0]) * 0.5))
            w_gap = int(round((self.output_shape[1] - img.shape[1]) * 0.5))
            new_image[h_gap:img.shape[0]+h_gap, w_gap:img.shape[1]+w_gap] = img[:, :]
            return new_image#process2.otsu_filter(new_image)
        img_list = []
        for boundary in self.objects_boundaries:
            object_pixel = self.objects[boundary]
            row_offset, col_offset = boundary[0], boundary[1]
            char_width = boundary[3] - boundary[1]
            char_height = boundary[2] - boundary[0]
            char_img = np.ones((char_height, char_width)) * self.background

            for x, y in object_pixel:
                char_img[x-row_offset, y-col_offset] = self.foreground
            char_img = char_image_add_margin(process2.normalise_scaling(char_img, output_shape=(self.output_shape[0]-offset,
                                                                                                self.output_shape[1]-offset)))
            #ImgIO.show_image(char_img)
            img_list.append(char_img)
        return img_list

    def segment_process(self, *args, **kwargs):
        self.vertical_segment()

    def get_ceiling_floor(self, object_list, axis=0):
        # axis = 0 : up, down
        # axis = 1 : left, right
        row_coor = [c[axis] for c in object_list]
        return min(row_coor), max(row_coor)

    def get_characters(self):
        if not self.objects:
            self.segment_process()
        if not self.char_images:
            self.char_images = self.make_characters_images()
        return self.char_images

    def sort_objects_left_boundary(self):
        self.objects_boundaries = sorted(self.objects_boundaries, key=lambda x: x[1])
        self.char_images = self.make_characters_images()

    def merge_two_objects(self, left_boundary, right_boundary):
        l_min_row, l_min_col, l_max_row, l_max_col = left_boundary
        r_min_row, r_min_col, r_max_row, r_max_col = right_boundary
        new_boundary = (l_min_row, l_min_col, r_max_row, r_max_col)
        self.objects_boundaries.remove(left_boundary)
        self.objects_boundaries.remove(right_boundary)
        self.objects_boundaries.append(new_boundary)
        tmp = self.objects[left_boundary]
        tmp.extend(self.objects[right_boundary])
        del self.objects[left_boundary]
        del self.objects[right_boundary]
        self.objects[new_boundary] = tmp



class SnakeSeparator(Separator):

    def __init__(self, image, output_shape=(70,70), foreground=None, length=-1, pixel_tolerance=1):
        self.image = image
        self.output_shape = output_shape
        self.objects = {}
        self.objects_boundaries = []
        self.char_images = []
        self.foreground = foreground if foreground else process2.get_background_color(image, foreground=True)
        self.background = process2.get_background_color(self.image, foreground=False)
        self.pixel_tolerance = pixel_tolerance
        self.snake_visited = []
        self.snake_lines = []

    def snake_segment(self, offset=5):
        height, width = self.image.shape[0], self.image.shape[1]
        initial_point, terminal_point = 0, height-1
        col_histogram = process2.pixels_histogram(self.image, axis=0, color=self.foreground)
        col_index = filter(lambda x:not (col_histogram[x] == 0), range(len(col_histogram)))
        left_boundary_col, right_boundary_col = col_index[0]-1, col_index[-1]+1
        start_boundary = map(lambda x:(x, left_boundary_col), range(height))
        end_boundary = map(lambda x:(x, right_boundary_col), range(height))
        directions_visited = [(1, 0), (0, 1), (0, -1), (-1, 0)]
        directions_reverse = [(-1, 0), (0, 1), (0, -1), (1, 0)]
        start, end = left_boundary_col+offset, right_boundary_col-offset
        def is_candidate(current, move, line, invalid):
            cur_x, cur_y = current
            next_x, next_y = cur_x+move[0], cur_y+move[1]
            if next_x < 0 or next_x >= height or next_y < 0 or next_y >= width or (next_x, next_y) in invalid :
                return False
            if not np.all(self.image[next_x, next_y] == self.foreground) and not (next_x, next_y) in line:
                return True
            else:
                return False
        def is_terminate(current, begin, initial, terminal):
            x, y = current
            return True if x == terminal or (x == initial and begin) else False
        def traverse(current, move, line):
            cur_x, cur_y = current
            next_ = cur_x+move[0], cur_y+move[1]
            line.append(next_)
            return next_
        def backtrack(current, move, line, invalid):
            cur_x, cur_y = current
            next_ = cur_x+move[0], cur_y+move[1]
            # if current in self.snake_visited:
            #     self.snake_visited.remove(current)
            #print "backtrack: %s" % str(next_)
            invalid.append(current)
            line.pop()
            return next_
        def snake_traverse(current, snake_line, invalid, directions):
            for move in directions:
                if is_candidate(current, move, invalid, snake_line):
                    if move == directions[-1]:
                        current = backtrack(current, move, snake_line, invalid)
                        return current
                    else:
                        current = traverse(current, move, snake_line)#, invalid)
                        return current
            else:
                current = snake_line[-1]
                invalid.append(current)
                snake_line.pop()
                return current
        def get_reverse_points(points):
            p = []
            for i in points:
                if p:
                    if i - 1 in p[-1]:
                        p[-1].append(i)
                    else:
                        p.append([i])
                else:
                    p.append([i])
            return [k[int((len(k)-1)*0.5)] for k in p]
        def get_split_line():
            for snake_start in range(start, end):
                begin_traverse = False
                current = (initial_point, snake_start)
                if np.all(self.image[current] == self.foreground):
                    continue
                else:
                    invalid_position = []
                    snake_line = [current]
                    while not is_terminate(current, begin_traverse, initial_point, terminal_point):
                        if not begin_traverse:
                            begin_traverse = True
                        current = snake_traverse(current, snake_line, invalid_position, directions_visited)
                    if current[0] == terminal_point:
                        self.snake_visited.extend(snake_line)
            self.snake_visited.extend(start_boundary)
            self.snake_visited.extend(end_boundary)
            end_points = [i for i in range(width) if (terminal_point, i) in self.snake_visited]
            reverse = get_reverse_points(end_points)
            for snake_reverse in reverse:
                begin_reverse = False
                current = (terminal_point, snake_reverse)
                invalid_position = []
                snake_line = [current]
                while not is_terminate(current, begin_reverse, terminal_point, initial_point):
                    if not begin_reverse:
                        begin_reverse = True
                    current = snake_traverse(current, snake_line, invalid_position, directions_reverse)
                if current[0] == initial_point:
                    #self.snake_visited.extend(snake_line)
                    self.snake_lines.append(snake_line)
        get_split_line()
        for index in range(len(self.snake_lines)):
            if index+1 >= len(self.snake_lines): break
            left, right = self.snake_lines[index], self.snake_lines[index+1]
            get_lower = lambda i: random.choice([p[1] for p in left if p[0] == i])
            get_upper = lambda i: random.choice([p[1] for p in right if p[0] == i])
            object_coordinates = [(i, j) for i in range(height) for j in range(
                    get_lower(i), get_upper(i)) if self.image[i, j] == self.foreground]
            ceiling, floor = self.get_ceiling_floor(object_coordinates, axis=0)
            lower, upper = self.get_ceiling_floor(object_coordinates, axis=1)
            boundary = (ceiling-1, lower-1, floor+1, upper+1)
            self.objects_boundaries.append(boundary)
            self.objects[boundary] = object_coordinates

    def segment_process(self, *args, **kwargs):
        self.snake_segment()
        self.sort_objects_left_boundary()

    def display_snake_segment(self, boundaries=None, interpolation=None):
        seg_image = self.image if self.image.ndim > 2 else process2.gray_to_rgb(self.image)
        if seg_image.max() <= 1:
            seg_image *= 255.
        target_color = np.array([0., 0., 255.])
        if not self.snake_visited:
            self.snake_segment()
        for pixel in self.snake_visited:
            seg_image[pixel] = target_color
        segment_color = np.array([255., 255., 0.])
        for line in self.snake_lines:
            for pixel in line:
                seg_image[pixel] = segment_color
        ImgIO.show_images_list([self.image, seg_image])

