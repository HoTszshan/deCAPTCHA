"""
Characters Segmentation...
"""
import os
import process
import random
import numpy
from lib import imgio as ImgIO
import time



class CharacterSeparator:

    def __init__(self, image, character_shape, length=4, foreground=(0, 0, 0)):
        self.image = process.gray_to_rgb(image) if image[0][0].size < 3 else image
        self.character_shape = character_shape
        self.length = length
        self.height = len(image)
        self.width =len(image[0])
        self.chunk_info_list = []

        self.foreground = foreground
        self.background = self.image[0][0].copy()
        self.__color_indices = [(r, g, b) for r in range(255, 0, -64) for g in range(255, 0, -64)
                for b in range(255, 0, -64) if r != g or r != b]

    def __count_pixels_per_line(self, axis=0, target_pixel=(0, 0, 0)):
        """
        Calculate the number of target pixel per line
        :param axis: 0 -> per col; else-> row
        :param target_pixel:
        :return:
        """
        number_list = []
        if axis == 0: # per col
            for i in range(self.width):
                count = len([1 for j in range(self.height) if self.__is_foreground_pixel(self.image[j,i],target_pixel)])
                number_list.append(count)
        else: # per row
            for j in range(self.height):
                count = len([1 for i in range(self.width) if self.__is_foreground_pixel(self.image[j,i], target_pixel)])
                number_list.append(count)
        return number_list

    def __color_fill_objects(self, object_pixel_list,
                           row, col, target_pixel=(0, 0, 0), filling_color=(255, 255, 0)):
        if self.__is_foreground_pixel(self.image[row][col], target_pixel):
            object_pixel_list.append((row, col))
            self.set_pixel_color(self.image[row][col], filling_color)
            directions = [(i, j) for i in range(-1, 2) for j in range(-1, 2) if not (i == 0 and j == 0)]
            for x, y in directions:
                if 0 <= row+x < self.height and 0 <= col+y < self.width:
                    self.__color_fill_objects(object_pixel_list, row+x, col+y, target_pixel, filling_color)
        else:
            pass

    def __circle_detection(self, pixel_list):
        pass
        """
        TODO
        """

    def __pixel_count_checking(self, pixel_list, min_count=50):
        return True if len(pixel_list) < min_count else False

    def __relative_position_checking(self, pixel_list):
        pass
        """
        TODO
        """

    def __even_cut(self, pixel_list, number):
        left_most, right_most = self.get_object_boundary(pixel_list)
        even_length = (right_most - left_most) / float(number)
        digits_list = {}.fromkeys(numpy.linspace(0, number-1, number, dtype=numpy.int))
        for i in digits_list.keys(): digits_list[i] = []
        for pixel in pixel_list:
            width = pixel[1] - left_most
            index = int(width / even_length) if width / even_length < number else number - 1
            digits_list[index].append(pixel)
        for i in digits_list.keys():
            changed_color = random.choice(self.__color_indices)
            for row, col in digits_list[i]:
                self.set_pixel_color(self.image[row][col], changed_color)
        return digits_list.values()

    def __merge_2_objects(self, objects_list):
        obj = objects_list.pop()
        color = self.image[objects_list[0][0][0], objects_list[0][0][1]]
        for row, col in obj:
            self.set_pixel_color(self.image[row][col], color)
        objects_list[0].extend(obj)

    def __merge_small_objects(self, object_list):
        def merge_objects(olist, cur_index, next_index):
            olist[cur_index].extend(olist[next_index])
            olist.remove(olist[next_index])
            changed_color = random.choice(self.__color_indices)
            for row, col in olist[cur_index]:
                self.set_pixel_color(self.image[row][col], changed_color)

        object_list = self.sort_objects(object_list)
        pixel_num = [len(obj) for obj in object_list]
        min_index = numpy.argmin(numpy.array(pixel_num))
        if 0 < min_index < len(object_list)-1:
            left, right = pixel_num[min_index-1], pixel_num[min_index+1]
            if left < right:
                merge_objects(object_list, min_index-1, min_index)
            else:
                merge_objects(object_list, min_index, min_index+1)
        elif min_index == 0:
            merge_objects(object_list, min_index, min_index+1)
        elif min_index == len(object_list)-1:
            merge_objects(object_list, min_index-1, min_index)
        return object_list

    def __is_foreground_pixel(self, current_pixel, target_pixel):
        if current_pixel[0] == target_pixel[0] and current_pixel[1] == target_pixel[1] and current_pixel[2] == target_pixel[2]:
            return True
        else:
            return False

    def __post_bounding_box_estimation(self, pixel_list):
        pass
        """
        TODO
        """

    def convert_object_to_img(self, pixel_list):
        left_most, right_most = self.get_object_boundary(pixel_list)
        character_img = numpy.ones((self.height, right_most-left_most+5)) * 255.0
        for row, col in pixel_list:
            character_img[row][col-left_most+2] = 0.0
        return character_img

    def convert_object_to_norm_img(self, pixel_list, norm_width, norm_height):
        up_most, down_most, left_most, right_most = self.get_object_all_boundary(pixel_list)
        character_height = down_most - up_most + 7
        character_width = right_most - left_most + 7
        if character_width < 18:  # too narrow
            character_width += 24
            character_img = numpy.ones((character_height, character_width)) * 255.0
            for row, col in pixel_list:
                character_img[row-up_most+3][col-left_most+15] = 0.0
        else:
            character_img = numpy.ones((character_height, character_width)) * 255.0
            for row, col in pixel_list:
                character_img[row-up_most+3][col-left_most+3] = 0.0
        character_img = process.filter_scale(character_img, norm_width, norm_height)
        return character_img

    def set_pixel_color(self, current_pixel, target_pixel):
        for i in range(len(current_pixel)):
            current_pixel[i] = target_pixel[i]

    def get_chunks_boundaries(self):
        return [chunk['boundary'] for chunk in self.chunk_info_list]

    def get_objects_number(self):
        return sum([len(chunk['objects']) for chunk in self.chunk_info_list])

    def get_objects_list(self):
        obj_list = []
        img_list = []
        for chunk in self.chunk_info_list:
            obj_list.extend(chunk['objects'])
        obj_list = self.sort_objects(obj_list)
        for obj in obj_list:
            img = self.convert_object_to_norm_img(obj, self.character_shape[1], self.character_shape[0])
            img_list.append(img)
        return img_list

    def get_object_boundary(self, pixel_list):
        col_list = [pixel[1] for pixel in pixel_list]
        return (min(col_list),  max(col_list))

    def get_object_all_boundary(self, pixel_list):
        row_list = [pixel[0] for pixel in pixel_list]
        col_list = [pixel[1] for pixel in pixel_list]
        return (min(row_list), max(row_list), min(col_list),  max(col_list))

    def detect_objects_in_chunk(self, chunk):
        for j in range(self.height):
            for i in range(chunk['boundary'][0], chunk['boundary'][1]):
                if self.__is_foreground_pixel(self.image[j, i], self.foreground):
                    pixel_list = []
                    self.__color_fill_objects(pixel_list, j, i, self.foreground, random.choice(self.__color_indices))
                    chunk['objects'].append(pixel_list)

        return chunk

    def color_filling_segmentation(self):
        for chunk in self.chunk_info_list:
            self.detect_objects_in_chunk(chunk)
        else:
            return True

    def vertical_segmentation(self, tolerance=3):
        col_hist = self.__count_pixels_per_line(axis=0) # per col
        # print self.__count_pixels_per_line(axis=1) # per row

        split_list = []
        flag_start = False
        for index in range(len(col_hist)):
            if col_hist[index] != 0 and not flag_start:
                split_list.append(index)
                flag_start = True
            if flag_start and col_hist[index] == 0:
                split_list.append(index)
                flag_start = False
        if not split_list:
            split_list.append(0)
            split_list.append(self.width)
        elif len(split_list) % 2 == 1:
            split_list.append(self.width)

        #chunk_list = []
        for index in range(0, len(split_list), 2):
            upper = split_list[index]
            lower = split_list[index + 1]
            if  self.chunk_info_list and upper - self.chunk_info_list[-1]['boundary'][1] < tolerance:
                pre_upper = self.chunk_info_list[-1]['boundary'][0]
                self.chunk_info_list[-1]['boundary'] = (pre_upper, lower)
                continue
            else:
                chunk_info_dict = {'boundary':(upper, lower),'objects':[]}
                self.chunk_info_list.append(chunk_info_dict)

        return True if self.chunk_info_list else False

    def thick_stuff_removal(self):
        for chunk in self.chunk_info_list:
            for pixel_list in chunk['objects']:
                """
                TODO: Detect stuffs
                """
                if self.__pixel_count_checking(pixel_list):
                    while pixel_list:
                        row, col = pixel_list.pop()
                        self.set_pixel_color(self.image[row, col],self.background)
            while [] in chunk['objects']: chunk['objects'].remove([])
        self.chunk_info_list = [chunk for chunk in self.chunk_info_list if len(chunk['objects']) > 0]

    def sort_objects(self, objects_list):
        objects_position = [(self.get_object_boundary(obj), obj) for obj in objects_list]
        objects_position = sorted(objects_position, key=lambda x: x[0][0])
        objects_list = [obj_position[1] for obj_position in objects_position]
        return objects_list

    def show_split_chunks(self):
        new_img = self.image.copy()
        chunk_list = self.get_chunks_boundaries()
        if chunk_list:
            split_cols = [index for num in range(1, len(chunk_list))
                 for index in range(chunk_list[num-1][1]+1, chunk_list[num][0]-1)]
            for j in range(self.height):
                for i in split_cols:
                    new_img[j, i, 0] = 255.0
                    new_img[j, i, 1] = 0.0
                    new_img[j, i, 2] = 0.0
        ImgIO.show_img(new_img, mode=1, title_name='Separated Chunks')

    def show_split_objects(self):
        img_list = self.get_objects_list()
        ImgIO.show_img_list(img_list)

    def check_objects(self, tolerance=3, min_pixel_num=300):
        even_length = self.width / self.length
        if self.get_objects_number() < self.length:
            # Even cut
            for chunk in self.chunk_info_list:
                for obj in chunk['objects']:
                    l, r = self.get_object_boundary(obj)
                    #print "obj width:", r - l
                    if r - l >= even_length * 1.8:
                        objs = self.__even_cut(obj, self.length - self.get_objects_number() + 1)
                        chunk['objects'].remove(obj)
                        chunk['objects'].extend(objs)
                        self.check_objects()
                        break
                    elif r - l >= even_length * 1.1:
                        objs = self.__even_cut(obj, 2)
                        chunk['objects'].remove(obj)
                        chunk['objects'].extend(objs)
                        self.check_objects()
                        break
        elif self.get_objects_number() > self.length:
            # Merge
            for chunk in self.chunk_info_list:
                chunk_width = chunk['boundary'][1] - chunk['boundary'][0]
                if len(chunk['objects']) == 2 and (chunk_width <= even_length or
                                                   abs(chunk_width - even_length) <= tolerance):
                    # chunk_width * 0.9<= even_length:
                    self.__merge_2_objects(chunk['objects'])
                elif len(chunk['objects']) > 2:
                    if min([len(obj) for obj in chunk['objects']]) < min_pixel_num:
                        chunk['objects'] = self.__merge_small_objects(chunk['objects'])
                        self.check_objects()
            else:
                if self.get_objects_number() > self.length:
                    self.merge_all_chucks()

                    self.check_objects()

        else:
            # Check length
            for chunk in self.chunk_info_list:
                for obj in chunk['objects']:
                    l, r = self.get_object_boundary(obj)
                    if r - l >= even_length * 1.37:
                        objs = self.__even_cut(obj, 2)
                        chunk['objects'].remove(obj)
                        chunk['objects'].extend(objs)
                        self.check_objects()
                        break

    def check_chucks(self, even_length):
        def remove_chunk(chunk_info_list, cur_index, next_index):
            l_left, l_right = chunk_info_list[cur_index]['boundary']
            r_left, r_right = chunk_info_list[next_index]['boundary']
            chunk_info_list[cur_index]['boundary'] = (l_left, r_right)
            chunk_info_list[cur_index]['objects'].extend(chunk_info_list[next_index]['objects'])
            chunk_info_list.remove(chunk_info_list[next_index])
            return chunk_info_list
        while len(self.chunk_info_list) > self.length:
            chuck_width = [chuck['boundary'][1] - chuck['boundary'][0] for chuck in self.chunk_info_list]
            min_index = numpy.argmin(numpy.array(chuck_width))
            if 0 < min_index < self.length-1: #print "middle"
                left_width, right_width = chuck_width[min_index-1], chuck_width[min_index+1]
                if left_width < right_width:
                    self.chunk_info_list = remove_chunk(self.chunk_info_list, min_index-1, min_index)
                else:
                    self.chunk_info_list = remove_chunk(self.chunk_info_list, min_index, min_index+1)
            elif min_index == 0:
                self.chunk_info_list = remove_chunk(self.chunk_info_list, min_index, min_index+1)
            elif min_index == self.length-1:
                self.chunk_info_list = remove_chunk(self.chunk_info_list, min_index-1, min_index)
        chuck_width = numpy.array([(lambda x: x[1] - x[0])(chuck['boundary']) for chuck in self.chunk_info_list])
        small_chuck = [i for i in range(len(self.chunk_info_list)) if (chuck_width < even_length / 2.0)[i]]
        while small_chuck:
            chuck_index = small_chuck.pop()
            if chuck_index == 0:
                self.chunk_info_list = remove_chunk(self.chunk_info_list, chuck_index, chuck_index+1)
            else:
                self.chunk_info_list = remove_chunk(self.chunk_info_list, chuck_index-1, chuck_index)
            chuck_width = numpy.array([(lambda x: x[1] - x[0])(chuck['boundary']) for chuck in self.chunk_info_list])
            small_chuck = [i for i in range(len(self.chunk_info_list)) if (chuck_width < even_length / 2.0)[i]]

    def merge_all_chucks(self):
        def remove_chunk(chunk_info_list, cur_index, next_index):
            l_left, l_right = chunk_info_list[cur_index]['boundary']
            r_left, r_right = chunk_info_list[next_index]['boundary']
            chunk_info_list[cur_index]['boundary'] = (l_left, r_right)
            chunk_info_list[cur_index]['objects'].extend(chunk_info_list[next_index]['objects'])
            chunk_info_list.remove(chunk_info_list[next_index])
            return chunk_info_list
        while len(self.chunk_info_list) > 1:
            self.chunk_info_list = remove_chunk(self.chunk_info_list, 0, 1)

    def segment_process(self):
        self.vertical_segmentation()
        self.color_filling_segmentation()
        self.thick_stuff_removal()
        self.check_chucks(even_length=(self.width / self.length))
        self.check_objects()
        #self.show_split_chunks()
        #self.show_split_objects()
        return self.get_objects_list()

    def save_segment_result(self, folder, label, sys_split='\\'):
        height, width = self.character_shape
        width *= self.length
        image = process.filter_scale(process.rgb_to_gray(self.image.copy()), width, height)
        for img in self.get_objects_list():
            image = numpy.hstack((image, numpy.zeros((height, 3)), img))
        image = numpy.hstack((image, numpy.zeros((height, 3))))

        if not os.path.exists(folder):
            os.makedirs(folder)
        file_list = [str(f).split('_')[0] for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        file_list = [file_name for file_name in file_list if file_name == label]
        save_file_path = folder + sys_split + label + '_' + str(len(file_list)) + '.jpg'
        ImgIO.write_img(image, save_file_path)

