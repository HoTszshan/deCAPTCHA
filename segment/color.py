import lib.process
import numpy
import sys
from projection import *
from sklearn.cluster import KMeans
from feature import process2
from segment import path
import time

sys.setrecursionlimit(10000)

# The input image should be binary image
class ColorFillingSeparator:

    def __init__(self, image, character_shape=(70,70), length=4, min_count=10, min_width=25):
        self.image = lib.process.gray_to_rgb(image) if image[0][0].size < 3 else image
        if image.max() <= 1.0:
            self.image = self.image * 255
        #process2.gray_to_rgb(image) if image.ndim < 3 else image
        self.character_shape = character_shape
        self.length = length
        self.height = len(image)
        self.width =len(image[0])
        self.chunk_info_list = []

        self.foreground = process2.get_background_color(self.image,foreground=True)
        self.background = process2.get_background_color(self.image)
        self.__color_indices = [(r, g, b) for r in range(255, 0, -64) for g in range(255, 0, -64)
                for b in range(255, 0, -64) if r != g or r != b]
        self.min_count = min_count
        self.min_width = min_width

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

    def __pixel_count_checking(self, pixel_list):
        return True if len(pixel_list) < self.min_count else False

    def __relative_position_checking(self, pixel_list):
        pass
        """
        TODO
        """

    def __even_cut(self, pixel_list, number, mode='BIDF'):
        if mode == 'even':
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

        else: # mode == DF
            tmp_image = np.zeros((self.image.shape[0], self.image.shape[1]))
            for x, y in pixel_list:
                tmp_image[x, y] = 1
            df = path.DropFallSeparator(tmp_image, foreground=1, length=number, mode='BIDF', B=1)
            # ImgIO.show_image(tmp_image)
            tmp, values = [], []
            left_most, right_most = self.get_object_boundary(pixel_list)
            df.split_num_objects(pixel_list, number, tmp, left_most, right_most, min_count=1)
            # print len(tmp), number
            # values = [v[1] for v in tmp if not len(v[1]) == 0]
            for v in tmp:
                if v[1]:
                    values.append(v[1])
                else:
                    continue
                left_most, right_most = self.get_object_boundary(v[1])
                if right_most - left_most > (self.width / self.length) * 1.2:
                    return self.__even_cut(pixel_list, number, mode='even')
            if len(values) == number:
                return values
            else:
                return self.__even_cut(pixel_list, number, mode='even')
            # split_index = df.get_min_split_index(pixel_list, foreground=1)
            # split_path = df.drop_fall_path(split_index)
            # l, r = df.split_object_by_path(pixel_list, split_path)
            # if len(l[1]) == 0 or len(r[1]) == 0:
            #     l, r = df.split_object_by_index(pixel_list, split_index)
            # if len(l[1]) == 0 or len(r[1]) == 0:
            #     return self.__even_cut(pixel_list, number, mode='even')
            # return [l[1], r[1]]


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
        character_img = numpy.ones((self.height, right_most-left_most+5)) * self.background[0]
        for row, col in pixel_list:
            character_img[row][col-left_most+2] = self.foreground[0]
        return character_img

    def convert_object_to_norm_img(self, pixel_list, norm_width, norm_height):
        up_most, down_most, left_most, right_most = self.get_object_all_boundary(pixel_list)
        character_height = down_most - up_most + 7
        character_width = right_most - left_most + 7
        if character_width < self.min_width:  # too narrow
            # character_width += 24
            # character_img = numpy.ones((character_height, character_width)) * self.background[0]
            # for row, col in pixel_list:
            #     character_img[row-up_most+3][col-left_most+15] = self.foreground[0]
            tmp = self.convert_object_to_img(pixel_list)
            # ImgIO.show_images_list([tmp, process2.normalise_scaling(tmp, (self.character_shape[1]-10, self.character_shape[0]-10))])
            tmp = process2.normalise_scaling(tmp, (self.character_shape[1]-10, self.character_shape[0]-10))
            # ImgIO.show_image(tmp)
            if not tmp.shape ==  (self.character_shape[1]-10, self.character_shape[0]-10):
                # print tmp.shape, (self.character_shape[1]-10, self.character_shape[0]-10)
                ImgIO.show_image(tmp)
                tmp = lib.process.filter_scale(tmp, self.character_shape[1] - 10, self.character_shape[0] - 10)
            character_img = numpy.ones(self.character_shape) * self.background[0]
            character_img[5:-5, 5:-5] = tmp[:, :]
            # ImgIO.show_images_list([tmp, character_img])
        else:
            character_img = numpy.ones((character_height, character_width)) * self.background[0]
            for row, col in pixel_list:
                character_img[row-up_most+3][col-left_most+3] = self.foreground[0]
            character_img = lib.process.filter_scale(character_img, norm_width, norm_height)
            # ImgIO.show_image(character_img)

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
        ImgIO.show_image(new_img, title_name='Separated Chunks')

    def show_split_objects(self):
        img_list = self.get_objects_list()
        ImgIO.show_images_list(img_list)

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
                    self.check_objects()
                    break
                elif len(chunk['objects']) > 2:
                    if min([len(obj) for obj in chunk['objects']]) < min_pixel_num:
                        chunk['objects'] = self.__merge_small_objects(chunk['objects'])
                        self.check_objects()
                        break
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
            if 0 < min_index < len(self.chunk_info_list)-1: #print "middle"
                left_width, right_width = chuck_width[min_index-1], chuck_width[min_index+1]
                if left_width < right_width:
                    self.chunk_info_list = remove_chunk(self.chunk_info_list, min_index-1, min_index)
                else:
                    self.chunk_info_list = remove_chunk(self.chunk_info_list, min_index, min_index+1)
            elif min_index == 0:
                self.chunk_info_list = remove_chunk(self.chunk_info_list, min_index, min_index+1)
            elif min_index == len(self.chunk_info_list)-1:
                self.chunk_info_list = remove_chunk(self.chunk_info_list, min_index-1, min_index)
        chuck_width = numpy.array([(lambda x: x[1] - x[0])(chuck['boundary']) for chuck in self.chunk_info_list])
        small_chuck = [i for i in range(len(self.chunk_info_list)) if (chuck_width < even_length / 2.0)[i]]
        while small_chuck:
            chuck_index = small_chuck.pop()
            if chuck_index == 0:
                self.chunk_info_list = remove_chunk(self.chunk_info_list, chuck_index, chuck_index+1)
            elif chuck_index == len(chuck_width) -1:
                self.chunk_info_list = remove_chunk(self.chunk_info_list, chuck_index-1, chuck_index)
            else:
                if chuck_width[chuck_index-1] > chuck_width[chuck_index+1]:
                    self.chunk_info_list = remove_chunk(self.chunk_info_list, chuck_index-1, chuck_index)
                else:
                    self.chunk_info_list = remove_chunk(self.chunk_info_list, chuck_index, chuck_index+1)
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
        # self.show_split_chunks()
        self.color_filling_segmentation()
        # self.show_split_chunks()
        self.thick_stuff_removal()
        # self.show_split_chunks()
        self.check_chucks(even_length=(self.width / self.length))
        try:
            self.check_objects()
        except RuntimeError:
            if self.get_objects_number() == self.length:
                return self.get_objects_list()
            else:
                self.check_objects()
                return self.get_objects_list()
        # self.show_split_chunks()
        # self.show_split_objects()
        return self.get_objects_list()

    def save_segment_result(self, folder, label):
        height, width = self.character_shape
        width *= self.length
        threshold = 254 if self.background[0]==255  else 1
        image = lib.process.filter_scale(process2.threshold_filter(process2.rgb_to_gray(self.image), threshold=threshold), width, height)
        # ImgIO.show_images_list([self.image, process2.rgb_to_gray(self.image), image])
        if image.max() <= 1.0:
            image = image * 255
        # print image.max(), self.foreground, self.background
        for img in self.get_objects_list():
            image = numpy.hstack((image, numpy.ones((height, 3))*self.foreground[0], img))
        image = numpy.hstack((image, numpy.ones((height, 3))*self.foreground[0]))
        # ImgIO.show_image(image)
        if not os.path.exists(folder):
            os.makedirs(folder)
        file_list = [str(f).split('_')[0] for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        file_list = [file_name for file_name in file_list if file_name == label]
        save_file_path = os.path.join(folder, label + '_' + str(len(file_list)) + '.jpg')
        ImgIO.write_image(image, save_file_path)

    def get_characters(self):
        return self.segment_process()


# TODO: to be improved
class ColorClusterSeparator(Separator):

    def __init__(self, image, output_shape=(70,70), n_cluster=16, foreground=None, length=4, pixel_tolerance=0):
        self.image = process2.rgb_to_gray(image) if image.ndim < 2 else image
        self.output_shape = output_shape
        self.n_cluster = n_cluster
        self.objects = {}
        self.objects_boundaries = []
        self.char_images = []
        self.foreground = np.zeros((1, 1, self.image.shape[2])) #foreground if foreground else process2.get_background_color(image, foreground=True)
        self.background = np.ones((1, 1, self.image.shape[2])) * 255. #process2.get_background_color(self.image, foreground=False)
        self.pixel_tolerance = pixel_tolerance
        self.length = length

    def kmeans_segment(self):
        height, width = self.image.shape[0], self.image.shape[1]
        start_time = time.time()
        new_image = self.image.copy()
        color_col_dict = self.image.reshape((self.image.shape[0]*self.image.shape[1]), self.image.shape[2])
        kmeans = KMeans(n_clusters=self.n_cluster, random_state=0).fit(np.array(color_col_dict))

        color_map = {}
        color_ = {}
        for label, color in zip(kmeans.predict(kmeans.cluster_centers_), kmeans.cluster_centers_):
            color_[str(label)] = color
        # print color_
        for i in range(height):
            for j in range(width):
                label = kmeans.predict(self.image[i, j])
                if str(label) in color_map.keys():
                    color_map[str(label)].append((i, j))
                else:
                    color_map[str(label)] = [(i, j)]
                new_image[i, j] = color_[str(label[0])]
        for obj in color_map.values():
            if len(obj) < (height * width )* 0.5 and len(obj) > 10 :
                ceiling, floor = self.get_ceiling_floor(obj, axis=0)
                lower, upper = self.get_ceiling_floor(obj, axis=1)
                boundary = (ceiling-1, lower-1, floor+1, upper+1)
                # if floor - ceiling <= height*0.66:
                if upper - lower < width * 0.95 :
                #     self.objects_boundaries.append(boundary)
                #     self.objects[boundary] = obj
                    if boundary in self.objects.keys():
                        self.objects[boundary].extend(obj)
                    else:
                        self.objects_boundaries.append(boundary)
                        self.objects[boundary] = obj
        finish_time = time.time()
        print "It takes %.4f to kmeans cluster " % (finish_time - start_time)
        # ImgIO.show_images_list([self.image, new_image])

        # Add position
        # color_col_dict = [np.array([i, j,self.image[i, j, 0], self.image[i, j,1], self.image[i, j,2]])
        #                   for i in range(height) for j in range(width) if not np.all(self.image[i,j]== self.background)]
        # kmeans = KMeans(n_clusters=self.n_cluster, random_state=0).fit(np.array(color_col_dict))
        # color_map = {}
        # color_ = {}
        # new_image = self.image.copy()#np.ones(self.image.shape)
        # for l, c in enumerate(kmeans.cluster_centers_):
        #     color_[l] = c[:2]
        # for value in color_col_dict:
        #     i, j = int(value[0]), int(value[1])
        #     label = kmeans.predict([value])
        #     # print label[0] , color_[label[0]]
        #     new_image[i, j] = self.image[int(color_[label[0]][0]), int(color_[label[0]][1])]
        #     # print new_image[i, j], self.image[i,j]
        #     if str(label) in color_map.keys():
        #         color_map[str(label)].append((i, j))
        #     else:
        #         color_map[str(label)] = [(i,j)]
        # for obj in color_map.values():
        #     if len(obj) < (height * width) * 0.5:
        #         boundary = self.get_boundary(obj)
        #         # if floor - ceiling <= height*0.66:
        #         self.objects_boundaries.append(boundary)
        #         self.objects[boundary] = obj
        #         # else:
        #         #     for coor in obj:
        #         #         new_image[coor] = self.background
        # finish_time = time.time()
        # print "It takes %.4f to kmeans cluster " % (finish_time - start_time)
        # # for index, value in self.objects.items():
        # #     print index, '\t', len(value), '\t',  value
        # ImgIO.show_images_list([self.image, new_image])
        self.image = new_image

        return new_image

    def remove_noise(self):
        tmp, new_objects = [], []
        for num in range(len(self.objects_boundaries)):
            tmp_image = np.ones(self.image.shape) * self.background
            tmp_boundary = self.objects_boundaries[num]
            tmp_object = self.objects[tmp_boundary]
            tmp.append(tmp_boundary)
            del self.objects[tmp_boundary]
            if len(tmp_object) > (self.image.shape[0] * self.image.shape[1]) * 0.5:
                continue
            for coor in tmp_object:
                tmp_image[coor] = self.foreground# 0# self.image[coor]
            cmp_image = tmp_image
            tmp_image = process2.inverse(process2.otsu_filter(tmp_image))
            # tmp_image = process2.reconstruction(process2.inverse(process2.otsu_filter(tmp_image)))

            if tmp_boundary[2] - tmp_boundary[0] > self.image.shape[0]*0.7:
                tmp_image = process2.reconstruction(tmp_image)
                # process.filter_remove_dots(process2.inverse(process2.otsu_filter(tmp_image)))
            # ImgIO.show_images_list([cmp_image, tmp_image], title_names=[str(tmp_boundary), str(tmp_image.sum())])
            if tmp_image.sum() > 50:
                new_object = [(i, j) for i in range(tmp_image.shape[0]) for j in range(tmp_image.shape[1]) if tmp_image[i, j]]
                new_boundary = self.get_boundary(new_object)
                new_objects.append((new_boundary, new_object))
        for b in tmp: self.objects_boundaries.remove(b)
        for b, obj in new_objects:
            self.objects_boundaries.append(b)
            self.objects[b] = obj

    def coarse_segment(self, max_object_width=40, min_object_count=30):
        def get_split_object(object_list, left, right):
            new_coordinates = []
            for row, col in object_list:
                if col >= left and col <= right:
                    new_coordinates.append((row, col))
            if new_coordinates:
                new_boundary = self.get_boundary(new_coordinates)
                return (new_boundary, new_coordinates)
            else:
                return None
        def split_images_to_objects(obj_list, new_list, min_obj_count, max_width):
            col_index, col_hist = zip(*self.get_object_projection_histogram(obj_list, get_index=True))
            derivative = [(col_hist[i]-col_hist[max(i-1,0)]) for i in range(len(col_hist))]
            if np.array(derivative).std() < 1: return None
            start, end = 0, len(col_hist)-1
            for index in range(len(col_index)):
                if col_hist[min(len(col_index)-1, index+1)] - col_hist[index] >=5:
                    start= col_index[index]
                    break
            for index in range(len(col_index)-1, 0, -1):
                if col_hist[max(0, index-1)] - col_hist[index] >= 5:
                    end = col_index[index]
                    break
            if start >= end: start, end = end, start
            new_obj = get_split_object(obj_list, start, end)
            new_list.append(new_obj)
        tmp_delete, new_objects = [], []
        for boundary in self.objects_boundaries:
            current_object= self.objects[boundary]
            ceiling, lower, floor, upper = boundary
            obj_width = upper - lower
            if len(self.objects[boundary]) > (self.image.shape[0]*self.image.shape[1])*0.5:
                tmp_delete.append(boundary)
                continue
            if obj_width > max_object_width:
                tmp_delete.append(boundary)
                # Vertical segment
                col_histogram = self.get_object_projection_histogram(current_object)
                boundary_list = []
                start_flag = False
                for index in range(len(col_histogram)):
                    if col_histogram[index] > self.pixel_tolerance:
                        if col_histogram[min(index+1, len(col_histogram)-1)] > self.pixel_tolerance and not start_flag:
                            boundary_list.append(lower+index)
                            start_flag = True
                            continue
                        elif col_histogram[min(index+1, len(col_histogram)-1)] <= self.pixel_tolerance and start_flag:
                            boundary_list.append(lower+index)
                            start_flag = False
                            continue
                if not len(boundary_list) % 2  == 0:
                    boundary_list.append(lower+len(col_histogram))
                if len(boundary_list) == 2:
                    split_images_to_objects(current_object, new_objects, min_object_count, max_object_width)
                    continue
                for index in range(0, len(boundary_list), 2):
                    left, right = boundary_list[index], boundary_list[index+1]
                    new_obj = get_split_object(current_object, left, right)
                    if not new_obj or len(new_obj[1]) < min_object_count or new_obj[0] in self.objects_boundaries:
                        continue
                    else:
                        new_objects.append(new_obj)
        for t in tmp_delete:
            self.objects_boundaries.remove(t)
            del self.objects[t]
        new_objects = filter(lambda x:x, new_objects)
        for b, obj in new_objects:
            self.objects_boundaries.append(b)
            self.objects[b] = obj


    def check_object_boundary(self):
        if len(self.objects_boundaries) > self.length:
            to_delete = []
            image_height, image_width = self.image.shape[0], self.image.shape[1]
            for boundary in self.objects_boundaries:
                ceiling, lower, floor, upper = boundary
                if floor - ceiling >= image_height * 0.85 or upper - lower >= image_width * 0.5:
                    to_delete.append(boundary)
            for b in to_delete:
                self.objects_boundaries.remove(b)
                del self.objects[b]

    def check_object_position(self, offset=1):
        tmp = []
        for boundary in self.objects_boundaries:
            ceiling, lower, floor, upper = boundary
            if ceiling < offset and floor < self.image.shape[0]:
                tmp.append(boundary)
        for b in tmp:
            self.objects_boundaries.remove(b)
            del self.objects[b]

    def post_processing(self):
        imgs = self.get_characters()
        # imgs = map(process2.median, imgs)
        imgs = map(process2.inverse, imgs)
        imgs = map(process.filter_fix_broken_characters, imgs)
        # ImgIO.show_images_list(imgs)
        self.char_images = imgs#map(process2.normalise_rotation, imgs)

    def check_number_count(self, min_num_count=100):
        if len(self.objects_boundaries) > self.length:
            to_delete = []
            for boundary, obj in self.objects.items():
                if len(obj) <= min_num_count:
                    to_delete.append(boundary)
            for boun in to_delete:
                self.objects_boundaries.remove(boun)
                del self.objects[boun]

    def remove_arc(self):
        if len(self.objects_boundaries) > self.length:
            to_delete = []
            for boundary, obj in self.objects.items():
                col_hist = self.get_object_contour_histogram(obj)
                derivative = [(col_hist[i]-col_hist[max(i-1,0)]) for i in range(len(col_hist))]
                if np.array(derivative).std() < 1:
                    to_delete.append(boundary)
            for boun in to_delete:
                self.objects_boundaries.remove(boun)
                del self.objects[boun]

    def remove_min_stuff(self):
        while len(self.objects_boundaries) > self.length:
            # ImgIO.show_images_list(self.get_characters())
            # self.display_kmeans_result()
            obj_count_num = [(boundary, len(obj)) for boundary, obj in self.objects.items()]
            obj_count_num = sorted(obj_count_num, key=lambda x:x[1])
            min_obj_boundary = obj_count_num[0][0]
            self.objects_boundaries.remove(min_obj_boundary)
            del self.objects[min_obj_boundary]
            # ImgIO.show_images_list(self.get_characters())
            # self.display_kmeans_result()

    def segment_process(self, verbose=False, *args, **kwargs):
        self.kmeans_segment()
        if verbose: self.display_kmeans_result()
        self.remove_noise()
        if verbose:  self.display_kmeans_result()
        self.coarse_segment(max_object_width=self.image.shape[1]/4)
        if verbose:  self.display_kmeans_result()
        self.check_object_boundary()
        if verbose:  self.display_kmeans_result()
        self.check_object_position()
        if verbose:  self.display_kmeans_result()
        self.check_number_count()
        self.remove_arc()
        self.remove_min_stuff()

        self.sort_objects_left_boundary()
        self.post_processing()
        # self.display_kmeans_result()

    def display_kmeans_result(self, boundaries=None, interpolation=None):
        number = len(self.objects_boundaries)
        #figsize=(10,10))#num='astronaut', figsize=(8,8))
        plt.figure()
        for num in range(number):
            plt.subplot(number+1, 1, num+1)
            title_name = str(self.objects_boundaries[num]) + str(len(self.objects[self.objects_boundaries[num]]))#"Image %s" % (num+1)
            plt.title(title_name)
            tmp_image = np.ones(self.image.shape) #* 255
            for coor in self.objects[self.objects_boundaries[num]]:
                tmp_image[coor] = self.image[coor]#1.0
            plt.imshow(tmp_image, interpolation=interpolation)
            plt.axis('off')
        plt.subplot(number+1, 1, number+1)
        plt.imshow(self.image)
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
        image = process2.rgb_to_gray(image)
        #initial = process2.max_min_normalization(process2.rgb_to_gray(self.image), max_value=1.0, min_value=0.0)
        #process2.resize_transform(initial, output_shape=(height, width))
        self.foreground = 255.
        for img in self.char_images:
            image = np.hstack((image, np.ones((height, 3))*self.foreground, img))
        image = np.hstack((image, np.zeros((height, 3))))
        # ImgIO.show_image(image)
        filename = dataset.get_save_image_name(folder, label, img_type='jpg')
                                               #+'_segment_result', img_type='jpg')
        ImgIO.write_image(image, filename)