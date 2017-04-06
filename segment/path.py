from projection import *

class SnakeSeparator(Separator):

    def __init__(self, image, output_shape=(70,70), foreground=None, length=-1, pixel_tolerance=1,):
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
            boundary = self.get_boundary(object_coordinates)
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


class DropFallSeparator(Separator):

    def __init__(self, image, output_shape=(70,70), foreground=None, length=4,
                 pixel_tolerance=1, method='projection', axis=0, alpha=1,
                 even_length=0, min_count=10, mode='BIDF', B=1):
        self.image = image
        self.output_shape = output_shape
        self.objects = {}
        self.objects_boundaries = []
        self.char_images = []
        self.foreground = foreground if foreground else process2.get_background_color(image, foreground=True)
        self.background = process2.get_background_color(self.image, foreground=False)
        self.pixel_tolerance = pixel_tolerance
        self.method = method
        self.axis = axis
        self.length = length
        self.alpha = alpha
        if even_length <= 0:
            lower, upper = self.get_chuck_position(self.image)
            self.even_length = round((upper - lower) / 4.)
        else:
            self.even_length = even_length
        self.min_count = min_count
        self.mode = mode
        self.beta = B
        self.__set_DF_direction_dict()
        self.__set_IDF_direction_dict()
        self.__set_BIDF_direction_dict()

    def get_chuck_position(self, image):
        col_hist = process2.pixels_histogram(image, axis=0, foreground=True)
        lower, upper = 0, len(col_hist)-1
        for i in range(len(col_hist)):
            if col_hist[i] > 0 and lower == 0:
                lower = i
                break
        for j in range(len(col_hist)-1, 0, -1):
            if col_hist[j] > 0 and upper == len(col_hist)-1:
                upper = j
                break
        return (lower, upper)

    def __set_DF_direction_dict(self):
        self.__DF_direction_dict = {1: (0, -1),
                                 2: (0, +1),
                                 3: (+1,+1),
                                 4: (+1, 0),
                                 5: (+1,-1),
        }

    def __set_IDF_direction_dict(self):
        self.__IDF_direction_dict = {(1, 0): (0, -1),
                                     (2, 0): (0, +1),
                                     (3, 0): (+1,+1),
                                     (6, 0): (+1, 0),
                                     (5, 0): (+1,-1),
                                     (4, +1):(+1, +1),
                                     (4, 0): (+1, 0),
                                     (4, -1): (+1, 0),
        }

    def __set_BIDF_direction_dict(self):
        self.__BIDF_direction_dict = {(1, 0): (0, -1),
                                 (2, 0): (0, +1),
                                 (3, 0): (+1,+1),
                                 (6, 0): (+1, 0),
                                 (5, 0): (+1,-1),
                                 (4, +1):(0, +1),
                                 (4, 0): (+1, 0),
                                 (4, -1): (+1, 0),
        }

    def __cal_DF_weight(self, image, x, y):
        weights = [image[x+i, y+j] * n for n, (i, j) in self.__DF_direction_dict.items()]
        if sum(weights) == 0 or sum(weights) == 15:
            return 4
        else:
            return max(weights)

    def __cal_IDF_weight(self, image, x, y, v_i):
        weights = [image[x+i, y+j] * n for n, (i, j) in self.__DF_direction_dict.items()]
        if sum(weights) == 0:
            return (4, v_i)
        elif sum(weights) == 15:
            return (6, 0)
        else:
            return (max(weights), 0)

    def __cal_BIDF_weight(self, image, x, y, B, v_i):
        if B == 0:
            return self.__cal_IDF_weight(image, x, y, v_i)

        area = [max(image[x+1, y-B+i:y+B+i+1]) for i in [-1, 0, 1]]
        area.extend([image[x, y+B+1], image[x, y-B-1]])
        weights = [ v*w for v, w in zip(area, range(5,0, -1))]
        if sum(weights) == 0:
            return (4, v_i)
        elif sum(weights) == 15:
            return (6, 0)
        else:
            return (max(weights), 0)

    def __is_terminate(self, image, x, y):
        if y == self.image.shape[1]-1 or y == 0:
            return True
        elif x == self.image.shape[0]-1:
            return True
        else:
            return False

    # # The background should be white(1)
    def drop_fall_path(self, start_index): #, method='gravity'):
        if self.mode == 'DF':
            def get_direction(img, x, y, path, v_i, B):
                dire = self.__DF_direction_dict[self.__cal_DF_weight(img, x, y)]
                if dire == self.__DF_direction_dict[2]:
                    if len(direction) >= 2:
                        pre1 = direction[-1]
                        pre2 = direction[-2]
                        if pre1 == self.__DF_direction_dict[1] and pre2 == self.__DF_direction_dict[2]:
                            dire = self.__DF_direction_dict[3]
                return dire
        elif self.mode == 'IDF':
            def get_direction(img, x, y, path, v_i, B):
                dire = self.__IDF_direction_dict[self.__cal_IDF_weight(img, x, y, v_i)]
                if dire == self.__DF_direction_dict[2]:
                    if len(direction) >= 2:
                        pre1 = direction[-1]
                        pre2 = direction[-2]
                        if pre1 == self.__IDF_direction_dict[(1, 0)] and pre2 == self.__IDF_direction_dict[(2, 0)]:
                            dire = self.__IDF_direction_dict[(3,0)]
                return dire
        elif self.mode == 'BIDF':
            def get_direction(img, x, y, path, v_i, B):
                dire = self.__BIDF_direction_dict[self.__cal_BIDF_weight(img, x, y, B, v_i)]
                if dire == self.__DF_direction_dict[2]:
                    if len(direction) >= 2:
                        pre1 = direction[-1]
                        pre2 = direction[-2]
                        if pre1 == self.__BIDF_direction_dict[(1, 0)] and pre2 == self.__BIDF_direction_dict[(2, 0)]:
                            dire = self.__BIDF_direction_dict[(3,0)]
                return dire
        else:
            def get_direction(img, x, y, path, v_i, B):
                dire = self.__BIDF_direction_dict[self.__cal_BIDF_weight(img, x, y, B, v_i)]
                if dire == self.__DF_direction_dict[2]:
                    if len(direction) >= 2:
                        pre1 = direction[-1]
                        pre2 = direction[-2]
                        if pre1 == self.__BIDF_direction_dict[(1, 0)] and pre2 == self.__BIDF_direction_dict[(2, 0)]:
                            dire = self.__BIDF_direction_dict[(3,0)]
                return dire
        path, direction = [], []
        tmp_image = process2.inverse(process2.otsu_filter(self.image)) # The background should be white(1)
        pos_x, pos_y = 0, start_index
        while not self.__is_terminate(tmp_image, pos_x, pos_y):
            v_i = pos_y - path[-self.alpha][1] if len(path) > self.alpha else 0
            v_i = 1 if v_i > 0 else int(v_i/self.alpha)
            path.append((pos_x, pos_y))
            dire = get_direction(tmp_image, pos_x, pos_y, path, int(v_i), self.beta)
            direction.append(dire)
            i, j = dire
            pos_x += i
            pos_y += j
        else:
            if not (pos_x, pos_y) in path:
                path.append((pos_x, pos_y))
        return path

    def split_object_by_path(self, object_list, path):
        left, right = [], []
        for x, y in object_list:
            split_points_y = [p[1] for p in path if x == p[0]]
            if y <= min(split_points_y):
                left.append((x, y))
            elif y >= max(split_points_y):
                right.append((x, y))
            else:
                # print "On the Path point:", (x, y)
                left.append((x, y))
        left_boundary = self.get_boundary(left) if left else None
        right_boundary = self.get_boundary(right) if right else None
        return (left_boundary, left), (right_boundary, right)

    def split_num_objects(self, obj, num, new_list, lower, upper, min_count):
        split_position = np.linspace(lower, upper, num+1, dtype=np.uint8)
        split_index = np.array(split_position[1:-1])
        right = obj
        tmp = []
        for index in split_index:
            right_list = right[1] if type(right)==tuple else right
            split_path = self.drop_fall_path(index)
            # self.display_drop_fall_path(split_path)
            left, right = self.split_object_by_path(right_list, split_path)
            if len(left[1]) > min_count:
                tmp.append(left)
            if len(tmp) >= num-1: break
        if len(right[1]) > min_count:
            tmp.append(right)
        if (self.get_boundary(obj), obj) in tmp:
            tmp.remove((self.get_boundary(obj), obj))
            mid = int((lower+upper) * 0.5)
            left, right = self.split_object_by_index(obj, mid)
            tmp.append(left)
            tmp.append(right)
        new_list.extend(tmp)

    # TODO: check carefully
    def check_object_length(self):
        if len(self.objects) < self.length:
            # self.display_segment_result()
            to_add, to_delete = [], []
            for boundary in self.objects_boundaries:
                ceiling, lower, floor, upper = boundary
                boundary_width = upper - lower
                # print boundary, self.even_length
                if boundary_width > self.even_length * 1.7 and boundary_width < self.even_length * 2.55:
                    to_delete.append(boundary)
                    split_index = int((lower+upper)*0.5)
                    split_path = self.drop_fall_path(split_index)
                    # split_index = self.get_min_derivative_split_index(self.objects[boundary], foreground=self.foreground)
                    # self.display_drop_fall_patxh(split_path)
                    left, right = self.split_object_by_path(self.objects[boundary], split_path)
                    to_add.append(left)
                    to_add.append(right)
                    continue
                if boundary_width > self.even_length * 2.55 and boundary_width < self.even_length * 3.4:
                    # print 3
                    to_delete.append(boundary)
                    self.split_num_objects(self.objects[boundary], 3, to_add, lower, upper, self.min_count)
                    continue
                if boundary_width > self.even_length * 3.4:
                    # print 4
                    to_delete.append(boundary)
                    self.split_num_objects(self.objects[boundary], 4, to_add, lower, upper, self.min_count)
                    continue

                if len(self.objects[boundary]) > self.even_length*self.even_length*0.7:
                    to_delete.append(boundary)
                    split_index = int((lower+upper)*0.5)
                    split_path = self.drop_fall_path(split_index)
                    left, right = self.split_object_by_path(self.objects[boundary], split_path)
                    to_add.append(left)
                    to_add.append(right)
                    continue
            for d in to_delete:
                self.objects_boundaries.remove(d)
                del self.objects[d]
            for b, obj in to_add:
                self.objects_boundaries.append(b)
                self.objects[b] = obj
            # self.display_segment_result()
            # ImgIO.show_images_list(self.get_characters())
            self.check_object_length()
        elif len(self.objects) > self.length:
            self.sort_objects_left_boundary()
            obj_width = [bound[3] - bound[1] for bound in self.objects_boundaries]
            merge_index = random.choice([i for i in range(len(obj_width)) if obj_width[i] == min(obj_width)])
            if merge_index == 0:
                self.merge_two_objects(self.objects_boundaries[merge_index], self.objects_boundaries[merge_index+1])
            elif merge_index == len(obj_width) -1:
                self.merge_two_objects(self.objects_boundaries[merge_index-1], self.objects_boundaries[merge_index])
            else:
                # Find nearest one
                left_upper = self.objects_boundaries[merge_index-1][3]
                right_lower = self.objects_boundaries[merge_index+1][1]
                ceiling, lower, floor, upper = self.objects_boundaries[merge_index]
                if lower-left_upper < right_lower-upper:
                    self.merge_two_objects(self.objects_boundaries[merge_index-1], self.objects_boundaries[merge_index])
                else:
                    self.merge_two_objects(self.objects_boundaries[merge_index], self.objects_boundaries[merge_index+1])
            self.check_object_length()

    def segment_process(self, *args, **kwargs):
        self.vertical_segment()
        self.check_object_length()
        # self.display_segment_result()
        self.sort_objects_left_boundary()

    def display_drop_fall_path(self, path, interpolation=None):
        plt.figure(num='segment result', figsize=(8,8))

        plt_image = self.image * 255.0 if np.max(self.image) <= 1.0 else self.image
        plt.subplot(2, 1 ,1)
        plt.imshow(np.uint8(plt_image),  interpolation='nearest')
        if plt_image.ndim <= 2:
            plt.gray()
        path = np.array(path)
        x = path[:, 0]
        y = path[:, 1]
        plt.plot(y, x, 'b.', linewidth=2)
        plt.axis('equal')
        plt.axis('off')
        plt.subplot(2, 1, 2)
        plt.imshow(process2.inverse(process2.otsu_filter(self.image)))
        plt.axis('equal')
        plt.axis('off')
        plt.show()

