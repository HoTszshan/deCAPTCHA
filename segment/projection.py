from matplotlib import pyplot as plt
from lib import dataset
from lib import imgio as ImgIO
from lib import process
from feature import process2
import numpy as np
import os
import random


class Separator(object):

    def __init__(self, image, output_shape=(70,70), foreground=None, length=-1,
                 pixel_tolerance=1, width_tolerance=3,method='projection', axis=0):
        self.image = image
        self.output_shape = output_shape
        self.objects = {}
        self.objects_boundaries = []
        self.char_images = []
        self.foreground = foreground if foreground else process2.get_background_color(image, foreground=True)
        self.background = process2.get_background_color(self.image, foreground=False)
        self.pixel_tolerance = pixel_tolerance
        self.width_tolerance = width_tolerance
        self.method = method
        self.axis = axis

    def __call__(self, image, *args, **kwargs):
        self.image = image
        return self.segment_process(*args, **kwargs)

    def segment_process(self, *args, **kwargs):
        self.vertical_segment(method=self.method, axis=self.axis)
        self.check_object_merge_gap()
        self.sort_objects_left_boundary()
        self.char_images = self.make_characters_images()

    def vertical_segment(self, method='projection', axis=0):
        height = self.image.shape[0]
        if method == 'projection':
            hist_func = process2.pixels_histogram
        elif method =='contour':
            hist_func = process2.boundary_histogram
        else:
            hist_func = process2.pixels_histogram
        col_histogram = hist_func(self.image, axis=axis, color=self.foreground)
        boundary_list = []
        start_flag = False
        # for i in range(0, len(col_histogram), 10):
        #     for j in range(i, min(i+10, len(col_histogram)), 1):
        #         print "(%3d, %3d) \t" %(j, col_histogram[j]),
        #     print ''

        for index in range(len(col_histogram)):
            if col_histogram[index] >= self.pixel_tolerance:
                if col_histogram[min(index+1, len(col_histogram)-1)] > self.pixel_tolerance and not start_flag:
                    boundary_list.append(index)
                    start_flag = True
                    continue
                elif col_histogram[min(index+1, len(col_histogram)-1)] < self.pixel_tolerance and start_flag:
                    boundary_list.append(index)
                    start_flag = False
                    continue
        if not len(boundary_list) % 2  == 0:
            boundary_list.append(len(col_histogram))
        # print "Boundary:", boundary_list, '\n'
        for index in range(0, len(boundary_list), 2):
            lower, upper = boundary_list[index], boundary_list[index+1]
            object_coordinates = [(i, j) for i in range(height) for j in range(lower, min(upper+1, len(col_histogram)))
                                  if self.image[i, j] == self.foreground]
            #ceiling, floor = self.get_ceiling_floor(object_coordinates)
            boundary = self.get_boundary(object_coordinates)#(ceiling-1, lower-1, floor+1, upper+1)
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
        if not self.objects or not self.char_images:
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
            image = np.hstack((image, np.ones((height, 3))*self.foreground, img))
        image = np.hstack((image, np.zeros((height, 3))))
        # ImgIO.show_image(image)
        filename = dataset.get_save_image_name(folder, label, img_type='jpg')
                                               #+'_segment_result', img_type='jpg')
        ImgIO.write_image(image, filename)

    def make_characters_images(self):
        offset = int(round(self.output_shape[0] * 0.15))
        background = self.background if self.image.ndim == 2 else self.background.mean()
        foreground = self.foreground if self.image.ndim == 2 else self.foreground.mean()
        def char_image_add_margin(img):
            img_height, img_width = img.shape
            if img_height >= self.output_shape[0] or img_width >= self.output_shape[1] :
                img = process2.normalise_scaling(img, output_shape=(self.output_shape[0]-offset, self.output_shape[1]-offset))
            new_image = np.ones(self.output_shape) * background
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
            char_img = np.ones((char_height, char_width)) * background
            for x, y in object_pixel:
                char_img[x-row_offset-1, y-col_offset-1] = foreground
            # ImgIO.show_image(char_img)
            char_img = char_image_add_margin(process2.normalise_scaling(char_img,
                        output_shape=(self.output_shape[0]-offset,self.output_shape[1]-offset), background=background))
            #ImgIO.show_image(char_img)
            img_list.append(char_img)
        return img_list

    def sort_objects_left_boundary(self):
        self.objects_boundaries = filter(lambda x: x, self.objects_boundaries)
        self.objects_boundaries = sorted(self.objects_boundaries, key=lambda x: x[1])
        # self.char_images = self.make_characters_images()

    def check_object_merge_gap(self):
        self.sort_objects_left_boundary()
        remove_objects, new_objects = [], []
        for i in range(len(self.objects_boundaries)-1):
            if self.objects_boundaries[i] in remove_objects:
                continue
            c_ceiling, c_lower, c_floor, c_upper = self.objects_boundaries[i]
            n_ceiling, n_lower, n_floor, n_upper = self.objects_boundaries[i+1]
            gap = n_lower - c_upper
            if gap < self.width_tolerance:
                remove_objects.append(self.objects_boundaries[i])
                remove_objects.append(self.objects_boundaries[i+1])
                tmp_list = []
                tmp_list.extend(self.objects[self.objects_boundaries[i]])
                tmp_list.extend(self.objects[self.objects_boundaries[i+1]])
                tmp_boundary = self.get_boundary(tmp_list)
                new_objects.append((tmp_boundary, tmp_list))
        for item in remove_objects:
            self.objects_boundaries.remove(item)
            del self.objects[item]
        for boundary, object in new_objects:
            self.objects_boundaries.append(boundary)
            self.objects[boundary] = object


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

    def get_object_projection_histogram(self, object_list, axis=0, get_index=False, foreground=None):
        object_list = np.array(object_list)
        if axis == 0: # each column
            min_value, max_value = min(object_list[:, 1]), max(object_list[:, 1])
            histogram = np.zeros(max_value-min_value+1, dtype=np.int8)
            for i, j in object_list:
                histogram[j-min_value] += 1
        else:
        #elif axis == 1: # each row
            min_value, max_value = min(object_list[:, 0]), max(object_list[:, 0])
            histogram = np.zeros(max_value-min_value+1, dtype=np.int8)
            for i, j in object_list:
                histogram[i-min_value] += 1
        if get_index:
            return zip(np.linspace(min_value, max_value, max_value-min_value+1, dtype=np.uint8), histogram)
        else:
            return histogram

    def get_object_contour_histogram(self, object_list, axis=0, get_index=False, foreground=None):
        object_list = np.array(object_list)
        if axis == 0: # each column
            min_value, max_value = min(object_list[:, 1]), max(object_list[:, 1])
            col_coor = np.linspace(min_value, max_value, max_value-min_value+1, dtype=np.uint8)
            contour_count = {}.fromkeys(col_coor)
            histogram = np.zeros(max_value-min_value+1, dtype=np.int8)
            for coor in object_list:
                i, j = coor[0], coor[1]
                if contour_count[j]:
                    contour_count[j].append(i)
                else:
                    contour_count[j] = [i]
            for key, rows in contour_count.items():
                min_, max_ = min(rows), max(rows)
                histogram[key-min_value] = max_ - min_ + 1
            return zip(col_coor, histogram) if get_index else histogram
        else:
        #elif axis == 1: # each row
            min_value, max_value = min(object_list[:, 0]), max(object_list[:, 0])
            row_coor = np.linspace(min_value, max_value, max_value-min_value+1, dtype=np.uint8)
            contour_count = {}.fromkeys(row_coor)
            histogram = np.zeros(max_value-min_value+1, dtype=np.int8)
            for coor in object_list:
                i, j = coor[0], coor[1]
                if contour_count[i]:
                    contour_count[i].append(j)
                else:
                    contour_count[i] = [j]
            for key, cols in contour_count.items():
                min_, max_ = min(cols), max(cols)
                histogram[key-min_value] = max_ - min_ + 1
            return zip(row_coor, histogram) if get_index else histogram

    def merge_two_objects(self, left_boundary, right_boundary):
        tmp = []
        tmp.extend(self.objects[left_boundary])
        tmp.extend(self.objects[right_boundary])
        new_boundary = self.get_boundary(tmp)
        self.objects_boundaries.remove(left_boundary)
        self.objects_boundaries.remove(right_boundary)
        del self.objects[left_boundary]
        del self.objects[right_boundary]
        self.objects_boundaries.append(new_boundary)
        self.objects[new_boundary] = tmp

    def split_object_by_index(self, object_list, index, index_axis=0):
        lower, upper = [], []
        if index_axis == 0: # vertical split by index
            for i, j in object_list:
                if i <= index:
                    lower.append((i, j))
                else:
                    upper.append((i, j))
        else:
        #elif index_axis == 1: # horizontal split by index
            for i, j in object_list:
                if j <= index:
                    lower.append((i, j))
                else:
                    upper.append((i, j))
        lower_boundary = self.get_boundary(lower)
        upper_boundary = self.get_boundary(upper)
        return (lower_boundary, lower), (upper_boundary, upper)

    def get_boundary(self, object_list):
        if object_list:
            ceiling, floor = self.get_ceiling_floor(object_list, axis=0)
            lower, upper = self.get_ceiling_floor(object_list, axis=1)
            boundary = (ceiling-1, lower-1, floor+1, upper+1)
            return boundary
        else:
            return None

    def projection_minimum_segment(self, object_list, offset):
        col_index, col_hist = zip(*self.get_object_projection_histogram(object_list, get_index=True))
        minimuns, new_objects = [], []
        for i in col_index:
            lower, upper = max(col_index[0], i-1), min(col_index[-1], i+1)
            if  col_index[lower] - col_hist[i]  > offset and col_hist[upper] - col_hist[i] > offset:
                minimuns.append(i)
        right = (self.get_boundary(object_list), object_list)
        for i in range(len(minimuns)):
            left, right = self.split_object_by_index(right[1], minimuns[i])
            new_objects.append(left)
        new_objects.append(right)
        return new_objects

    def get_min_split_index(self, obj_list, axis=0, foreground=None, method='projection'):
        if method == 'contour':
            col_index, col_hist = zip(*self.get_object_contour_histogram(obj_list, axis=axis, foreground=foreground, get_index=True))
        else: # method == 'projection':
            col_index, col_hist = zip(*self.get_object_projection_histogram(obj_list, axis=axis,foreground=foreground, get_index=True))
        split_index = random.choice([col_index[i] for i in range(len(col_hist)) if col_hist[i] == min(col_hist)])
        return split_index

    def get_min_derivative_split_index(self,obj_list, axis=0, foreground=None, method='projection', get_index_list=False):
        if method == 'contour':
            col_index, col_hist = zip(*self.get_object_contour_histogram(obj_list, axis=axis, foreground=foreground, get_index=True))
        else: # method == 'projection':
            col_index, col_hist = zip(*self.get_object_projection_histogram(obj_list, axis=axis,foreground=foreground, get_index=True))
        derivative = [(col_hist[i]-col_hist[max(i-1,0)]) for i in range(len(col_hist))]
        if get_index_list:
            index_derivative = sorted(zip(col_index, derivative), key=lambda x:x[1])
            indices, derivatives = zip(*index_derivative)
            return list(indices)
        else:
            split_index = random.choice([col_index[i] for i in range(len(col_hist)) if derivative[i] == min(derivative)])
            return split_index






