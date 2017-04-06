"""
Character Recognition...
# Suppose that there are 100 sample edge points in each shape

"""
from skimage import feature as ski_feature
from lib import imgio as ImgIO
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance as spa_dist
import scipy.stats as stats
import numpy as np
import random
import math
import copy
import time


max_gray_level = 2 ** 8 - 1
dist2 = lambda point1, point2: math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

class ShapeContext:

    def __init__(self, image, sigma=3.0, patch=3, points_number=100):
        self.height = len(image)
        self.width = len(image[0])
        self.image = image
        self.gradient_shape()
        self.sample_edgepoints = self.__sampling_shape_edges(sigma,patch,points_number)
        self.calculate_mean_distance()
        self.calculate_shape_context(self.sample_edgepoints[0])
        self.sc_list = zip(self.sample_edgepoints,[self.calculate_shape_context(point)#self.calculate_generalized_shape_context(point)
                                                   for point in self.sample_edgepoints])
        #self.sc_dict = dict.fromkeys(self.sample_edgepoints)
        #for point in self.sc_dict.keys(): self.sc_dict[point] = self.calculate_shape_context(point)


    def __sampling_shape_edges(self, sigma, patch, number):
        shape_image = ski_feature.canny(self.image,sigma)
        #ImgIO.show_img(shape_image)
        get_min_dis_coor1 = lambda matrix: np.argmin(np.min(matrix, 1))
        return self.__get_sampling_3(shape_image, number, patch, get_min_dis_coor1)

    def __get_sampling_1(self, shape_img, number, patch):
        start_time = time.time()
        ratio = float(number) / float(shape_img.sum())
        point_list = []
        if ratio < 1:
            for row in range(0, self.height, patch):
                for col in range(0, self.width, patch):
                    tmp_list = [(row+j, col+i) for j in range(patch) for i in range(patch)
                            if 0 <= row+j < self.height and 0 <= col+i < self.width and shape_img[row+j,col+i]]
                    sample_num = int(math.ceil(len(tmp_list) * ratio))
                    while len(tmp_list) > sample_num:
                        tmp_list.remove(tmp_list[random.randint(0,len(tmp_list)-1)])
                    point_list.extend(tmp_list)
            bigger_patch = patch
            while len(point_list) > number:
                tmp_ratio = float(number) / len(point_list)
                update_point_list = []
                bigger_patch =  bigger_patch * patch if bigger_patch < self.width else bigger_patch
                for row in range(0, self.height, bigger_patch):
                    for col in range(0, self.width, bigger_patch):
                        tmp_list = [(i, j) for i, j in point_list if row <= i < row+bigger_patch and col <= j < col+bigger_patch]
                        new_sample_num = int(math.ceil(len(tmp_list) * tmp_ratio))
                        while len(tmp_list) > new_sample_num:
                            tmp_list.remove(tmp_list[random.randint(0,len(tmp_list)-1)])
                        update_point_list.extend(tmp_list)
                point_list = update_point_list
        else:
            point_list.extend([(row, col) for row in range(self.height)
                          for col in range(self.width) if shape_img[row, col]])
        end_time = time.time()
        print "Sampling 1 time :", end_time - start_time, "s"
        return point_list

    def __get_sampling_2(self, shape_img, number, nearest_coor):
        # Jitendra: unit length space sampling (for source matlab code)
        start_time = time.time()
        point_list = [(i, j) for j in range(self.width) for i in range(self.height) if shape_img[i][j]]
        random.shuffle(point_list)
        while len(point_list) > number:
            num = len(point_list)
            dis_mat = spa_dist.squareform(spa_dist.pdist(point_list))
            dis_mat += np.eye(num) * num ** 2
            point_list.remove(point_list[nearest_coor(dis_mat)])
        end_time = time.time()
        print "Sampling 2 time :", end_time - start_time, "s"
        return point_list

    def __get_sampling_3(self, shape_img, number, patch, nearest_coor):
        start_time = time.time()
        ratio = float(number) / sum(sum(shape_img))
        point_list = []
        if ratio < 1:
            for row in range(0, self.height, patch):
                for col in range(0, self.width, patch):
                    tmp_list = [(row+j, col+i) for j in range(patch) for i in range(patch)
                            if 0 <= row+j < self.height and 0 <= col+i < self.width and shape_img[row+j,col+i]]
                    sample_num = int(math.ceil(len(tmp_list) * ratio))
                    while len(tmp_list) > sample_num:
                        #tmp_list.remove(tmp_list[random.randint(0,len(tmp_list)-1)])
                        dis_mat = spa_dist.squareform(spa_dist.pdist(tmp_list)) + np.eye(len(tmp_list)) * len(tmp_list) ** 2
                        tmp_list.remove(tmp_list[nearest_coor(dis_mat)])
                    point_list.extend(tmp_list)
            while len(point_list) > number:
                num = len(point_list)
                dis_mat = spa_dist.squareform(spa_dist.pdist(point_list))
                dis_mat += np.eye(num) * num ** 2
                point_list.remove(point_list[nearest_coor(dis_mat)])
        else:
            point_list.extend([(row, col) for row in range(self.height)
                          for col in range(self.width) if shape_img[row, col]])
        end_time = time.time()
        #print "Sampling 3 time :", end_time - start_time, "s"
        return point_list

    def display_self_points(self, symbol='bx'):
        new_img = np.ones((self.height, self.width, 3)) * 255
        plt.imshow(np.uint8(new_img))
        col = [point[1] for point in self.sample_edgepoints]
        row = [point[0] for point in self.sample_edgepoints]
        plt.plot(col, row, symbol)
        plt.axis('off')
        plt.axis('equal')
        plt.show()

    def display_points_image(self):
        new_img = np.zeros((self.height,self.width),dtype=bool)
        for row, col in self.sample_edgepoints:
            new_img[row][col] = True
        ImgIO.show_img(new_img)

    def display_sampling(self, sigma=3.0, patch=3, number=100):
        shape_image = ski_feature.canny(self.image,sigma)
        #ImgIO.show_img(shape_image)
        get_min_dis_coor1 = lambda matrix: np.argmin(np.min(matrix, 1))
        p1 = self.__get_sampling_1(shape_image, number, patch)
        p2 = self.__get_sampling_2(shape_image, number, get_min_dis_coor1)
        p3 = self.__get_sampling_3(shape_image, number, patch, get_min_dis_coor1)
        self.__display_sampling(shape_image, p1, p2, p3)

    def __display_sampling(self, shape_image, points1, points2, points3):
        plt.figure()
        new_img = np.ones((self.height, self.width,3)) * 255.0

        plt.subplot(2, 2, 1)
        plt.gray()
        plt.title("Edges")
        plt.imshow(shape_image)
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.title("Sampling 1")
        plt.imshow(np.uint8(new_img))
        plt.gray()
        y1 = [point[1] for point in points1]
        x1 = [point[0] for point in points1]
        plt.plot(y1, x1, "b+")
        plt.axis('equal')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.title("Sampling 2")
        plt.imshow(np.uint8(new_img))
        plt.gray()
        y2 = [point[1] for point in points2]
        x2 = [point[0] for point in points2]
        plt.plot(y2, x2, "r.")
        plt.axis('equal')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.title("Sampling 3")
        plt.imshow(np.uint8(new_img))
        plt.gray()
        y3 = [point[1] for point in points3]
        x3 = [point[0] for point in points3]
        plt.plot(y3, x3, "r.")
        plt.axis('equal')
        plt.axis('off')
        plt.show()

    def calculate_shape_context(self, point, logr=5, theta=30):
        p_x, p_y = point
        bins = np.zeros((logr, int(360/theta)), dtype=np.uint8)
        for q_x, q_y in self.sample_edgepoints:
            if not (q_x == p_x and q_y == p_y):
                dis = math.sqrt(((q_x - p_x)/ float(self.height)) ** 2 + ((q_y - p_y)/float(self.width)) ** 2) / self.alpha
                angle_index = get_angle(p_x, p_y, q_x, q_y) / theta
                dis_index = int(math.floor(math.log(dis, 2))) + logr -1 if dis < math.log(logr-1, 2) else logr - 1
                bins[dis_index][angle_index] += 1
        #return [bins[j,i] for j in range(logr) for i in range(int(360/theta))]
        return [bins[j,i] / float(len(self.sample_edgepoints)-1)  for j in range(logr) for i in range(int(360/theta))]

    def gradient_shape(self):
        self.dx = self.image.copy()
        self.dy = self.image.copy()
        for x in range(self.height):
            for y in range(self.width):
                if x-1 >= 0 and y-1 >= 0:
                    self.dx[x][y] = (self.image[x][y] - self.image[x-1][y]) / 255.0
                    self.dy[x][y] = (self.image[x][y] - self.image[x][y-1]) / 255.0
                elif x == 0 and y== 0:
                    self.dx[x][y] = 0.0
                    self.dy[x][y] = 0.0
                elif x > 0:
                    self.dx[x][y] = (self.image[x][y] - self.image[x-1][y]) / 255.0
                    self.dy[x][y] = 0.0
                elif y > 0:
                    self.dx[x][y] = 0.0
                    self.dy[x][y] = (self.image[x][y] - self.image[x][y-1]) / 255.0

    def calculate_generalized_shape_context(self, point, logr=5, theta=30):
        p_x, p_y = point
        bins = np.zeros((int(360/theta), logr * 2),dtype=np.uint8)
        for q_x, q_y in self.sample_edgepoints:
            if not (q_x == p_x and q_y == p_y):
                dis = math.sqrt(((q_x - p_x)/ float(self.height)) ** 2 + ((q_y - p_y)/float(self.width)) ** 2) / self.alpha
                angle_index = get_angle(p_x, p_y, q_x, q_y) / theta
                dis_index = int(math.log(dis, 2)) + 3 if dis < logr-1 else logr - 1
                bins[angle_index][dis_index * 2] += self.dx[q_x][q_y]
                bins[angle_index][dis_index * 2 + 1] += self.dy[q_x][q_y]
        #return [bins[j,i] for j in range(logr) for i in range(int(360/theta))]
        if not bins.sum() == 0:
            bins = bins / float(bins.sum())
        return [bins[i][j] for j in range(logr * 2) for i in range(int(360/theta))]

    def update_points(self, points):

        def cut_off(val, axis=0):
            val = 0 if val < 0 else val
            if axis == 0: # x
                if val >= self.height: val = self.height - 1
            elif axis == 1: # y
                if val >= self.width: val = self.width - 1
            return val
        #self.sample_edgepoints = [(int(round(px)), int(round(py))) for px, py in points]
        self.sample_edgepoints = [(cut_off(int(round(px)),axis=0), cut_off(int(round(py)),axis=1)) for px, py in points]
        self.sc_list = zip(self.sample_edgepoints,[self.calculate_shape_context(point) #self.calculate_generalized_shape_context(point)#
                                                   for point in self.sample_edgepoints])
        #print self.sample_edgepoints

    def calculate_mean_distance(self):
        l2_dis = lambda x, y: math.sqrt(((y[0] - x[0]) / float(self.height)) ** 2 + ((y[1] - x[1]) / float(self.width)) ** 2)
        dist = [l2_dis(point_x, point_y) for point_x in self.sample_edgepoints for point_y in self.sample_edgepoints]
        self.alpha = sum(dist) / float(len(dist))

    def get_tangent_angle(self, point):
        x, y = point
        if  0 < x < self.height and 0 < y < self.width:
            dx = self.image[x][y] - self.image[x-1][y] #if 0 < x  else 0
            dy = self.image[x][y] - self.image[x][y-1] #if 0 < y < self.width else 0
            return math.atan2(dy, dx)
        else:
            return None

def get_angle(p_x, p_y, q_x, q_y):
    dx = q_x - p_x
    dy = q_y - p_y
    if dx == 0:
        angle = 0 if dy >= 0 else 180
    elif dx > 0:
        angle = np.arctan(float(dy)/float(dx)) / np.pi * 180
        angle = angle + 180 if angle < 0 else angle
    else:
        angle = np.arctan(float(dy)/float(dx)) / np.pi * 180
        angle = angle + 360 if angle < 0 else angle
    if dy == 0:
        angle = 90 if dx > 0 else 270
    return int(angle)

def hungarian_matching(cost_matrix):
    mat = cost_matrix.copy()

    def row_operation(m):
        for i in range(len(m)):
            min_element = min(m[i])
            for j in range(len(m[i])):
                m[i][j] -= min_element

    def col_operation(m):
        for j in range(len(m[0])):
            min_element = min(m[:,j])
            for i in range(len(m)):
                m[i][j] -= min_element

    def assignment(m):
        def remove_element_from_candidates(candidate_list, target_col):
            for i in range(len(candidate_list)):
                if target_col in candidate_list[i][1]:
                    candidate_list[i] = (candidate_list[i][0], [col for col in candidate_list[i][1] if not col == target_col])
            candidate_list = [(row, cols) for row, cols in candidate_list if not len(cols) == 0]
            return candidate_list


        def remove_row_from_candidates(candidate_list, target_row):
            for i, cols in candidate_list:
                if i == target_row:
                    candidate_list.remove((i, cols))
                    break
            return candidate_list

        matching_list = []
        candidates = [(i, [j for j in range(len(m[i])) if m[i][j] == 0 ]) for i in range(len(m))]
        while candidates:
            least_col_num = min([len(row[1]) for row in candidates])
            for row, col_candidates in candidates:
                if len(col_candidates) == least_col_num:
                    if least_col_num == 1:
                        matching_list.append((row,col_candidates[0]))
                        candidates = remove_element_from_candidates(candidates, col_candidates[0])
                        candidates = remove_row_from_candidates(candidates,row)
                    else:
                        selected_index = random.choice(col_candidates)
                        matching_list.append((row, selected_index))
                        candidates = remove_element_from_candidates(candidates,selected_index)
                        candidates = remove_row_from_candidates(candidates,row)
                    break

        return matching_list

    def draw_least_lines(m):
        row_lines = [False] * len(m)
        col_lines = [False] * len(m)
        for i in range(len(m)):
            zero_index_in_row = [j for j in range(len(m[i])) if m[i][j] == 0 and not col_lines[j]]
            if len(zero_index_in_row) > 1:

                row_lines[i] = True
            elif len(zero_index_in_row) == 1 and not col_lines[zero_index_in_row[0]]:
                col_lines[zero_index_in_row[0]] = True
        return (sum(row_lines) + sum(col_lines), row_lines, col_lines)

    def find_lowest_value(m,rows, cols):
        rest_values = [m[i][j] for i in range(len(m)) for j in range(len(m[i])) if not (rows[i] or cols[j])]
        return min(rest_values) if rest_values else 0

    def update_matrix(m, rows, cols, lowest):
        for i in range(len(m)):
            for j in range(len(m[i])):
                if rows[i] and cols[j]:
                    m[i][j] += lowest
                elif not (rows[i] or cols[j]):
                    m[i][j] -= lowest

    row_operation(mat)
    col_operation(mat)
    lines_number, row_lines, col_lines = draw_least_lines(mat)
    while lines_number < len(cost_matrix):
        lowest = find_lowest_value(mat, row_lines,col_lines)
        update_matrix(mat, row_lines, col_lines, lowest)
        lines_number, row_lines, col_lines = draw_least_lines(mat)
    else:
        matching_list =  assignment(mat)
        return  matching_list

def hungarian_assignment(cost_matix):
    row_ind, col_ind = linear_sum_assignment(cost_matix)
    return [(row_ind[i], col_ind[i]) for i in range(len(row_ind))]

class TPS_Morpher:
    dist_L2 = lambda self, point1, point2: math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    kernel_func = lambda self, r: r ** 2 * math.log(r ** 2) if not r == 0 else 0

    def __init__(self, points_o, points_t,  l_0=0):
    #def __init__(self, points_o, points_t, width,height, l_0=0):
        #self.width = width
        #self.height = height
        self.alpha = None
        self.l_0 = l_0
        self.points_O = [(x / 100.0, y / 100.0) for x, y in points_o]#points_o #[(x / float(self.height), y / float(self.width)) for x, y in points_o]
        self.points_T = [(x / 100.0, y / 100.0) for x, y in points_t]#points_t #[(x / float(self.height), y / float(self.width)) for x, y in points_t]
        #self.mat_K = None
        self.fx, self.fy = self.calculate_TPS_weights(self.points_O, self.points_T)
        #print self.fx
        #print self.fy

    def calculate_TPS_weights(self, points_A, points_B):
        p = len(points_A)
        if not self.alpha:
            self.alpha = sum([self.dist_L2(j, i) for j in points_A for i in points_A]) / float(len(points_A) * len(points_A))
        mat_I = np.mat(np.eye(len(points_A)))
        self.mat_K = np.mat([[self.kernel_func(self.dist_L2(points_A[i], points_A[j])) + mat_I[i,j] * self.alpha ** 2 * self.l_0
                              for j in range(p)] for i in range(p)])
        #for one in self.mat_K: print one
        mat_P = np.mat([(1, point[0], point[1]) for point in points_A])
        mat_O = np.mat(np.zeros((3, 3)))

        mat_L = np.mat(np.vstack((np.hstack((self.mat_K, mat_P)),np.hstack((mat_P.T, mat_O)))))

        vector_bx = np.concatenate([[points_B[i][0] for i in range(p)], np.zeros(3)])
        vector_by = np.concatenate([[points_B[i][1] for i in range(p)], np.zeros(3)])
        fx = np.linalg.solve(mat_L, vector_bx)#[val[0] for val in (mat_L.I * vector_bx)]
        fy = np.linalg.solve(mat_L, vector_by)#[val[0] for val in (mat_L.I * vector_by)]
        return (fx[-3], fx[-2], fx[-1], fx[:-3]) , (fy[-3], fy[-2], fy[-1], fy[:-3])

    def interpolate_z_value(self, a1, a2, a3, w, points_A, new_point):
        kernel_values = [w[i] * self.kernel_func(self.dist_L2(points_A[i], new_point)) for i in range(len(points_A))]
        return a1 + a2 * new_point[0] + a3 * new_point[1] + sum(kernel_values)

    def coordinate_transform(self, new_point):
        new_x = self.interpolate_z_value(self.fx[0], self.fx[1], self.fx[2], self.fx[3], self.points_O, new_point)
        new_y = self.interpolate_z_value(self.fy[0], self.fy[1], self.fy[2], self.fy[3], self.points_O, new_point)
        return (new_x, new_y)

    def update_target_points(self, points):
        tmp =  [self.coordinate_transform(p) for p in points]
        #return [(x * self.height, y * self.width) for x, y in tmp]
        return tmp

    def calculate_bending_energy(self):
        mat_w = np.vstack((self.fx[3], self.fy[3]))
        #for one in mat_w.T: print one

        Q = (mat_w * self.mat_K * mat_w.T).tolist()
        Q = [Q[i][j] for i in range(len(Q)) for j in range(len(Q[0]))]
        return np.sum(Q) / float(len(Q))
        #return I_fx + I_fy

    def define_target_points(self, points_t):
        self.points_T = [(x / 100.0, y / 100.0) for x, y in points_t]#points_t #[(x / float(self.height), y / float(self.width)) for x, y in points_t]
        #self.mat_K = None
        self.fx ,self.fy = self.calculate_TPS_weights(self.points_O, self.points_T)


class ShapeContextMatcher:

    def __init__(self, sc1, sc2):
        self.shape_context_1 = sc1
        self.shape_context_2 = sc2

        self.cost_matrix = self.__calculate_cost_matrix()
        #self.cost_matrix = self.__normalize_scale(self.cost_matrix)
        self.matching_pairs = hungarian_assignment(self.cost_matrix)#hungarian_matching(self.cost_matrix)
        #self.__TPS_model = None
        #self.__affine_model = None

    def __calculate_matching_gsc(self,shapecontext1,shapecontext2):
        if len(shapecontext1) == len(shapecontext2):
            return  spa_dist.pdist(np.array([shapecontext1, shapecontext2]))[0]
            #diff_cost = [(float(shapecontext1[i]) - float(shapecontext2[i])) ** 2 / (float(shapecontext1[i]) + float(shapecontext2[i]))
            #        for i in range(len(shapecontext1)) if shapecontext1[i] != shapecontext2[i]]
            #return 0.5 * sum(diff_cost)
        else:
            print 'Matching Points Error!', "sc_1 :", len(shapecontext1), "sc_2 :", len(shapecontext2)

    def __calculate_matching_sc(self,shapecontext1,shapecontext2):
        if len(shapecontext1) == len(shapecontext2):
            diff_cost = [(float(shapecontext1[i]) - float(shapecontext2[i])) ** 2 / (float(shapecontext1[i]) + float(shapecontext2[i]))
                         if not (shapecontext1[i] + shapecontext2[i]) == 0.0 else 0.0 for i in range(len(shapecontext1))]
            return 0.5 * sum(diff_cost)
        else:
            print 'Matching Points Error!', "sc_1 :", len(shapecontext1), "sc_2 :", len(shapecontext2)

    def __calculate_matching_tan(self, angle1, angle2):
        if angle1 and angle2:
            return 0.5 * (1 - math.cos(angle1 - angle2))
        else:
            return 9999.9

    def __calculate_matching_cost(self, sc1, sc2, e1, e2, beta=0.1):
        c_sc = self.__calculate_matching_sc(e1[1],e2[1])
        #c_tan = self.__calculate_matching_tan(sc1.get_tangent_angle(e1[0]), sc2.get_tangent_angle(e2[0]))
        return c_sc
        #return (1 - beta) * c_sc + beta * c_tan


    def __calculate_cost_matrix(self):
        if len(self.shape_context_1.sample_edgepoints) == len(self.shape_context_2.sample_edgepoints):
            cost_matrix = [[self.__calculate_matching_cost(self.shape_context_1, self.shape_context_2, element_1, element_2)
                             for element_2 in self.shape_context_2.sc_list] for element_1 in self.shape_context_1.sc_list]
            return np.asarray(cost_matrix)
        else:
            print 'Calculate Cost Matrix Error!'

    def __normalize_scale(self, matrix):
        mean_dis = matrix.sum()  / float(len(matrix) ** 2)
        return matrix / mean_dis

    def transformation_affine_model(self):
        calculate_vector = lambda p, q: (p[0] - q[0], p[1] - q[1])
        calculate_average = lambda s, num: (s[0] / float(num), s[1] / float(num))
        calculate_sum = lambda vectors: (sum(list(vectors[0])), sum(list(vectors[1])))

        points_1 = self.shape_context_1.sample_edgepoints
        points_2 = self.shape_context_2.sample_edgepoints

        offset_vectors = [calculate_vector(points_1[x], points_2[y]) for x, y in self.matching_pairs]
        o_vector = calculate_average(calculate_sum(zip(*offset_vectors)), len(self.matching_pairs))

        print o_vector

        coordinates = [(points_1[p_index], points_2[q_index]) for  p_index, q_index in self.matching_pairs]
        P, Q = zip(*coordinates)
        P = np.hstack((np.ones((len(self.matching_pairs), 1)), np.array([list(point) for point in P])))
        Q = np.hstack((np.ones((len(self.matching_pairs), 1)), np.array([list(point) for point in Q])))
        Q_pse = np.linalg.pinv(Q)
        A_mat = (np.mat(Q_pse) * np.mat(P)).T

        print A_mat


    def transformation_TPS_model(self,points_o, points_t, l_0=1, iter_num=1):
        #def interpolate_iterator(p_set1, p_set2, num):
        t_points = points_t
        tps_model = None
        for i in range(iter_num):
            if not tps_model:
                tps_model = TPS_Morpher(points_o, t_points, l_0)
            else:
                tps_model.define_target_points(t_points)
            #tps_model = TPS_Morpher(points_o, t_points,self.shape_context_1.width, self.shape_context_1.height, l_0)
            t_points = tps_model.update_target_points(points_o)
        return tps_model, t_points
        #return interpolate_iterator(points_o, points_t, iter_num)
        #return self.__TPS_model
        #return TPS_Morpher(points_2, points_1, l_0)

    def calculate_shape_context_distance(self, new_sc):
        new_cost_mat = np.array([[self.__calculate_matching_cost(self.shape_context_1, new_sc, e1, e2)
                                  for e2 in new_sc.sc_list] for e1 in self.shape_context_1.sc_list])
        n = len(new_cost_mat)
        m = len(new_cost_mat[0])

        min_row_list = [min(new_cost_mat[i, :]) for i in range(n)]#[np.argmin(new_cost_mat[i, :]) for i in range(n)]
        min_col_list = [min(new_cost_mat[:, i]) for i in range(m)]#[np.argmin(new_cost_mat[:, i]) for i in range(m)]
        sc_dis = sum(min_row_list) / float(n) + sum(min_col_list) / float(m)

        return sc_dis

    def calculate_appearance_cost(self,sc_p, sc_tq,tps, delta=3):
        half_size = delta / 2

        def get_image_window_indices(point, half_window_size):
            return [(point[0]+i, point[0]+j) for i in range(-half_window_size, half_window_size+1)
                    for j in range(-half_window_size, half_window_size+1)]

        def get_image_window_pixels(points, img, img_width, img_height):
            return [img[x, y] / max_gray_level if 0 <= x < img_height and 0 <= y < img_width else 0 for x, y in points]

        def update_tps_points(points):
            tmp = tps.update_target_points(points)
            return [(int(round(x)), int(round(y))) for x, y in tmp]


        def window_diff_squa(list1, list2, window_size):
            tmp = [(list1[i] - list2[i])**2 for i in range(len(list1))]
            return [[tmp[i*window_size +j] for j in range(window_size)]for i in range(window_size)]

        def get_gaussian_list(half_window_size):
            x = np.arange(-half_size, half_size+1, 1)
            y = stats.norm.pdf(x, 0, 1)
            y = y / np.sum(y)
            return y

        def gaussian_window_diff(window, gauss):
            row = len(window)
            col = len(window[0])

            for i in range(row):
                window[i, :] = np.multiply(window[i, :], gauss[i])
            for j in range(col):
                window[:, j] = np.multiply(window[:, j], gauss[j])

            return window.sum()

        def gaussian_window_points(pixels, window_size, gauss):
            patch = np.array([[pixels[i * window_size + j] for j in range(window_size)] for i in range(window_size)])
            for i in range(window_size):
                patch[i, :] = np.multiply(patch[i, :], gauss[i])
            for j in range(window_size):
                patch[:, j] = np.multiply(patch[:, j], gauss[j])
            return patch


        sum = 0
        gauss = get_gaussian_list(half_size)

        for i in range(len(sc_p.sample_edgepoints)):
            p_index = get_image_window_indices(sc_p.sample_edgepoints[i], half_size)
            q_index = get_image_window_indices(sc_tq.sample_edgepoints[i], half_size)
            q_index = update_tps_points(q_index)
            mat_p = get_image_window_pixels(p_index, sc_p.image, sc_p.width, sc_p.height)
            mat_q = get_image_window_pixels(q_index, sc_tq.image, sc_tq.width, sc_tq.height)
            p = gaussian_window_points(mat_p,delta,gauss)
            q = gaussian_window_points(mat_q,delta,gauss)
            diff = np.sum([(p-q)[i][j] ** 2 for j in range(delta) for i in range(delta)])#math.sqrt()
            #print diff
            sum += diff
        #print sum, "/",  len(sc_p.sample_edgepoints), "=", sum / float(len(sc_p.sample_edgepoints))
        ac = sum / float(len(sc_p.sample_edgepoints)) #* 0.001
        return ac


    def calculate_shape_distance(self, l_0=1, iter_num=5,ac=1.6, sc=1.0, be=0.3):
        points_1, points_2 = zip(*[(self.shape_context_1.sample_edgepoints[i], self.shape_context_2.sample_edgepoints[j])
                              for i, j in self.matching_pairs])
        #points_1 = [(x / float(self.shape_context_1.height), y / float(self.shape_context_1.width)) for x, y in points_1]
        #points_2 = [(x / float(self.shape_context_2.height), y / float(self.shape_context_2.width)) for x, y in points_2]
        tps, new_points = self.transformation_TPS_model(points_2, points_1,l_0=l_0, iter_num=iter_num)

        new_sc = copy.deepcopy(self.shape_context_2)
        new_sc.update_points(new_points)
        #self.shape_context_2.update_points(new_points)
        shape_context_distance = self.calculate_shape_context_distance(new_sc)

        appearance_cost = self.calculate_appearance_cost(self.shape_context_1, new_sc, tps)

        bending_energy = tps.calculate_bending_energy()
        #print appearance_cost, shape_context_distance, bending_energy #, shape_context_distance +  bending_energy * be
        #print appearance_cost*ac, shape_context_distance*sc, bending_energy*be #, shape_context_distance +  bending_energy * be
        return appearance_cost*ac + shape_context_distance*sc +bending_energy*be


    def calculate_easy_shape_distance(self):
        distances = [self.cost_matrix[row][col] for row,col in self.matching_pairs]
        """
        for i, j in self.matching_pairs:
            print [self.shape_context_1.sample_edgepoints[i], self.shape_context_2.sample_edgepoints[j]],
        print distances
        """
        return sum(distances)

    def display_two_shapes(self, symbol1='ro', symbol2='go'):
        new_img = np.ones((self.shape_context_1.height, self.shape_context_1.width,3)) * 255
        plt.imshow(np.uint8(new_img))
        col1 = [point[1] for point in self.shape_context_1.sample_edgepoints]
        row1 = [point[0] for point in self.shape_context_1.sample_edgepoints]
        plt.plot( col1, row1, symbol1)

        col2 = [point[1] for point in self.shape_context_2.sample_edgepoints]
        row2 = [point[0] for point in self.shape_context_2.sample_edgepoints]
        plt.plot(col2, row2, symbol2)

        plt.axis('off')
        plt.axis('equal')
        plt.show()

    def display_two_shapes_matching(self, symbol1='ro', symbol2='go'):
        new_img = np.ones((self.shape_context_1.height, self.shape_context_1.width,3)) * 255
        plt.imshow(np.uint8(new_img))
        col1 = [point[1] for point in self.shape_context_1.sample_edgepoints]
        row1 = [point[0] for point in self.shape_context_1.sample_edgepoints]
        plt.plot( col1, row1, symbol1)

        col2 = [point[1] for point in self.shape_context_2.sample_edgepoints]
        row2 = [point[0] for point in self.shape_context_2.sample_edgepoints]
        plt.plot( col2, row2, symbol2)
        #print len(self.matching_pairs)
        #check_matching_overlap(self.matching_pairs)

        for origin, target  in self.matching_pairs:
            x = [self.shape_context_1.sample_edgepoints[origin][1], self.shape_context_2.sample_edgepoints[target][1]]
            y = [self.shape_context_1.sample_edgepoints[origin][0], self.shape_context_2.sample_edgepoints[target][0]]
            plt.plot(x[:],y[:],'b:')

        plt.axis('off')
        plt.axis('equal')
        plt.show()

    def display_matching_processing(self,symbol1='ro', symbol2='go'):
        plt.figure()#num='astronaut', figsize=(8,8))
        """
        plt.subplot(1,4,1)
        plt.title("image 1")
        plt.imshow(np.uint8(self.shape_context_1.image))
        plt.gray()
        plt.axis('off')

        plt.subplot(1,4,2)
        plt.title("image 2")
        plt.imshow(np.uint8(self.shape_context_2.image))
        plt.gray()
        plt.axis('off')
        """
        #plt.subplot(1,4,3)
        plt.subplot(1,4,1)
        plt.title("shape 1")
        shape_image_1 = ski_feature.canny(self.shape_context_1.image,3)
        plt.imshow(np.uint8(shape_image_1))
        plt.gray()
        plt.axis('off')

        #plt.subplot(1,4,4)
        plt.subplot(1,4,2)
        plt.title("shape 2")
        shape_image_2 = ski_feature.canny(self.shape_context_2.image,3)
        plt.imshow(np.uint8(shape_image_2))
        plt.gray()
        plt.axis('off')
        #"""
        #plt.subplot(1,3,1)
        plt.subplot(1,4,3)
        plt.title("sample points 1")
        new_img_1 = np.ones((self.shape_context_1.height, self.shape_context_1.width,3)) * 255
        plt.imshow(np.uint8(new_img_1))
        col1 = [point[1] for point in self.shape_context_1.sample_edgepoints]
        row1 = [point[0] for point in self.shape_context_1.sample_edgepoints]
        plt.plot( col1, row1, symbol1)
        plt.axis('off')
        plt.axis('equal')

        #plt.subplot(1,3,2)
        plt.subplot(1,4,4)
        plt.title("sample points 2")
        new_img_2 = np.ones((self.shape_context_2.height, self.shape_context_2.width,3)) * 255
        plt.imshow(np.uint8(new_img_2))
        col2 = [point[1] for point in self.shape_context_2.sample_edgepoints]
        row2 = [point[0] for point in self.shape_context_2.sample_edgepoints]
        plt.plot( col2, row2, symbol2)
        plt.axis('off')
        plt.axis('equal')
        """
        #plt.subplot(1,3,3)
        plt.subplot(1,5,5)
        plt.title("matching")
        new_img = np.ones((55,40,3)) * 255
        plt.imshow(np.uint8(new_img))
        col1 = [point[1] for point in self.shape_context_1.sample_edgepoints]
        row1 = [point[0] for point in self.shape_context_1.sample_edgepoints]
        plt.plot( col1, row1, 'r.')

        col2 = [point[1] for point in self.shape_context_2.sample_edgepoints]
        row2 = [point[0] for point in self.shape_context_2.sample_edgepoints]
        plt.plot( col2, row2, 'gx')
        #print len(self.matching_pairs)
        check_matching_overlap(self.matching_pairs)

        for origin, target  in self.matching_pairs:
            x = [self.shape_context_1.sample_edgepoints[origin][1], self.shape_context_2.sample_edgepoints[target][1]]
            y = [self.shape_context_1.sample_edgepoints[origin][0], self.shape_context_2.sample_edgepoints[target][0]]
            plt.plot(x[:],y[:],'b:')

        plt.axis('off')
        plt.axis('equal')
        """
        plt.show()



## Mean Image
# file_list = [os.path.join(dir_path,f) for f in os.listdir(dir_path) if not f.endswith('.DS_Store')]
# img_list = [process.filter_threshold(process.filter_scale(imgio.read_img_uc(image),width=70,height=70), threshold=160) for image in file_list]
# aimg = process.filter_average_image(img_list)#process.filter_remove_confusion(process.filter_average_image(img_list), threshold=50)


# K-medoids
# total cost distance is least


class GeneralizedShapeContext:
    def __init__(self, image, sigma=3.0, sample='gsc', sample_params=100):
        self.sigma = sigma
        self.normalize_image(image)
        shape_image = self.get_contours()
        if sample == 'gsc':
            sample_points, self.norm_sample_points = self.sampling_edge_gsc(shape_image,
                                                        number=sample_params, nearest_coor=self.__get_nearest_coor)
            mat_dist, mat_angle = self.calculate_mat_distIndex_angleIndex(self.norm_sample_points)
            self.sc_dict = self.calculate_gsc(sample_points, self.norm_sample_points, mat_dist, mat_angle)
            self.rsc = None
        elif sample == 'rsc':
            sample_points, self.norm_sample_points = self.sampling_edge_gsc(shape_image,
                                                        number=100, nearest_coor=self.__get_nearest_coor)
            mat_dist, mat_angle = self.calculate_mat_distIndex_angleIndex(self.norm_sample_points)
            self.sc_dict = self.calculate_gsc(sample_points, self.norm_sample_points, mat_dist, mat_angle)
            self.r_points, self.rsc = self.calculate_rsc_1(self.sc_dict, self.norm_sample_points, ratio=sample_params)
        else:
            print "Sampling method is not define!"
            print "Sampling to generate generalized shape context!"
            sample_points, self.norm_sample_points = self.sampling_edge_gsc(shape_image,
                                                        number=100, nearest_coor=self.__get_nearest_coor)
            mat_dist, mat_angle = self.calculate_mat_distIndex_angleIndex(self.norm_sample_points)
            self.sc_dict = self.calculate_gsc(sample_points, self.norm_sample_points, mat_dist, mat_angle)
            self.rsc = None

    def normalize_image(self, image):
        self.height, self.width = image.shape
        self.image = image.copy().astype(float) / float(max_gray_level)
        """
        dx = copy.deepcopy(self.image)
        dy = copy.deepcopy(self.image)

        tmp_x = copy.deepcopy(dx)
        tmp_y = copy.deepcopy(dy)
        tmp_x[:, 1:] = dx[:, :(self.width - 1)]
        dx = dx - tmp_x
        #ImgIO.show_img(dx)
        tmp_y[1:, :] = dy[:(self.height - 1), :]
        dy = dy - tmp_y
        """
        dx, dy = np.gradient(self.image)
        #ImgIO.show_img(dy)
        self.dx, self.dy = dx, dy

    def calculate_mat_distIndex_angleIndex(self, norm_point_list, logr=5, theta=30):
        #point_list = point_list[:5]
        samp_num =len(norm_point_list)
        mat_dis = spa_dist.squareform(spa_dist.pdist(norm_point_list, metric='euclidean'))

        y_col = np.zeros((samp_num, 1))
        y_row = np.zeros((1, samp_num))
        y_col[:, 0] = norm_point_list[:, 1]
        y_row[0, :] = norm_point_list[:, 1]
        mat_dy = np.repeat(y_col, samp_num, axis=1) - np.repeat(y_row, samp_num, axis=0)
        x_col = np.zeros((samp_num, 1))
        x_row = np.zeros((1, samp_num))
        x_col[:, 0] = norm_point_list[:, 0]
        x_row[0, :] = norm_point_list[:, 0]
        mat_dx = np.repeat(x_col, samp_num, axis=1) - np.repeat(x_row, samp_num, axis=0)
        mat_theta = np.arctan2(mat_dy, mat_dx) * 180 / np.pi
        mat_theta = (mat_theta + 360) % 360
        mat_angle_index = np.floor(mat_theta / theta)

        try:
            self.mean_dist
        except AttributeError:
            self.mean_dist = np.mean(mat_dis)

        mat_dis = mat_dis / self.mean_dist
        mat_dis[mat_dis == 0] = 2 ** logr
        mat_dis_logr = np.floor(np.log2(mat_dis)) + (logr - 1)
        mat_dis_logr[mat_dis_logr < 0] = 0
        mat_dis_logr[np.all([mat_dis_logr>=logr, mat_dis_logr < 2 * logr -1], axis=0)] = logr - 1
        return mat_dis_logr, mat_angle_index

    def calculate_gsc(self, point_list, norm_point_list, mat_disI, mat_angI, logr=5, theta=30):
        gsc_dict = {}
        samp_num =len(point_list)
        ang_num = 360 / theta
        bins_num = logr * 2 * ang_num
        for i in range(samp_num):
            tmp_gsc = np.zeros(bins_num)
            #print mat_disI[:, i]
            for r in range(logr):
                r_indices = np.where(mat_disI[:, i]==r)[0]
                for j in r_indices:
                    if not i == j:
                        angle_index = int(mat_angI[j,i])
                        x, y = point_list[j]
                        tmp_gsc[ang_num * 2 * r + angle_index * 2] += self.dx[x][y]
                        tmp_gsc[ang_num * 2 * r + angle_index * 2 + 1] += self.dy[x][y]
                        #print (angle_index, self.dx[x][y], self.dy[x][y])
            #self.print_gsc(tmp_gsc)
            gsc_dict[i] = tmp_gsc
        return gsc_dict

    def calculate_rsc_1(self, gsc, norm_point_list, ratio=0.1):
        num_k = int(len(norm_point_list) * ratio)
        indices = random.sample(gsc.keys(), num_k)
        representative_points = []
        rsc = {}
        for i in range(num_k):
            representative_points.append(norm_point_list[indices[i]])
            rsc[i] = gsc[indices[i]]
        return np.array(representative_points), rsc

    def calculate_rsc_2(self, gsc, norm_point_list, ratio=0.1):
        num_k = int(len(norm_point_list) * ratio)
        patch = self.width / num_k
        points_img = np.zeros((self.height, self.width), dtype=bool)
        points = (norm_point_list * np.array([self.width, self.height]))
        for x, y in points:
            points_img[int(x)][int(y)] = True
        r_points, r_norm_points = self.sampling_edge_gsc(points_img, number=num_k,
                                nearest_coor=self.__get_nearest_coor, patch=patch)
        """
        is_rep = np.zeros(len(norm_point_list), dtype=bool)
        for point_x, point_y in r_norm_points:
            for i in range(len(norm_point_list)):
                if norm_point_list[i][0] == point_x and norm_point_list[i][1] == point_y:
                #if norm_point_list[i] == point:
                    is_rep[i] = True
                    break
        ind = [i for i in range(len(is_rep)) if is_rep[i]]
        """
        indices = [i for i in range(len(norm_point_list)) for point_x, point_y in r_norm_points
                   if norm_point_list[i][0] == point_x and norm_point_list[i][1] == point_y]
        rsc = {}
        for i in range(num_k):
            rsc[i] = gsc[indices[i]]
        return np.array(r_points), rsc


    def display_r_points(self, points, symbol='rx'):
        new_img = np.ones((self.height, self.width,3)) * 255
        plt.imshow(np.uint8(new_img))
        col = [self.get_y_coordinate(point[1]) for point in points]
        row = [self.get_x_coordinate(point[0]) for point in points]
        plt.plot( col, row, symbol)
        plt.axis('off')
        plt.axis('equal')
        plt.show()

    def print_gsc(self, tmp_gsc, logr=5, ang_num=12):
        for r_ in range(logr):
            for t_ in range(0,2*ang_num,2):
                print '(', tmp_gsc[2*ang_num*r_ + t_], tmp_gsc[2*ang_num*r_ + t_+1], ')' ,
            print ''

    def sampling_edge_gsc(self, shape_img, number, nearest_coor, patch=3):
        ratio = float(number) / float(shape_img.sum())
        point_list = []
        if ratio < 1:
            for row in range(0, self.height, patch):
                for col in range(0, self.width, patch):
                    tmp_list = [[row+j, col+i] for j in range(patch) for i in range(patch)
                            if 0 <= row+j < self.height and 0 <= col+i < self.width and shape_img[row+j,col+i]]
                    sample_num = int(math.ceil(len(tmp_list) * ratio))
                    while len(tmp_list) > sample_num:
                        tmp_list.remove(tmp_list[random.randint(0, len(tmp_list)-1)])
                        #dis_mat = spa_dist.squareform(spa_dist.pdist(tmp_list)) + np.eye(len(tmp_list)) * len(tmp_list) ** 2
                        #tmp_list.remove(tmp_list[nearest_coor(dis_mat)])
                    point_list.extend(tmp_list)
            while len(point_list) > number:
                num = len(point_list)
                dis_mat = spa_dist.squareform(spa_dist.pdist(point_list))
                dis_mat += np.eye(num) * num ** 2
                point_list.remove(point_list[nearest_coor(dis_mat)])
        else:
            point_list.extend([[row, col] for row in range(self.height)
                          for col in range(self.width) if shape_img[row, col]])

        normalized_point_list = [[self.get_x_normalized(x), self.get_y_normalized(y)] for x, y in point_list]
        return np.array(point_list), np.array(normalized_point_list)

    def get_contours(self):
        return ski_feature.canny(self.image, self.sigma)

    get_x_normalized = lambda self, x: float(x) / float(self.height)

    get_y_normalized = lambda self, y: float(y) / float(self.width)

    get_x_coordinate = lambda self, x: int(x * self.height)

    get_y_coordinate = lambda self, y: int(y * self.width)

    __get_nearest_coor = lambda self, matrix: np.argmin(np.min(matrix, 1))


class FastPruner:

    def __init__(self, label_list, gsc_list):
        self.shape_size = len(gsc_list)
        self.gsc_dimension = (gsc_list[0].gsc[0].shape)[0]
        self.shape_gsc_num = len(gsc_list[0].gsc)
        self.shape_dict = self.initialize_shape_dict(label_list, gsc_list)


    def initialize_shape_dict(self, label_list, gsc_list):
        tmp_dict = {}
        for i in range(len(label_list)):
            tmp_dict[i] = gsc_list[i]
        return tmp_dict


    def calculate_gsc_distance_1(self, gsc_1, gsc_2):
        dist = sum((gsc_1 - gsc_2) ** 2)
        return math.sqrt(dist)

    def calculate_gsc_distance_2(self, gsc_1, gsc_2):
        numerator = (gsc_1 - gsc_2) ** 2
        denominator = np.fabs(gsc_1) + np.fabs(gsc_2)
        return sum([numerator[i] / denominator[i] if not denominator[i] == 0 else 0 for i in range(len(gsc_1))])


    def calculate_voting_distance(self, mat_Gi, num_u, num_si, num_gsc, metric='l2_dis'):
        metric_func_name = ['l2_dis', 'chi_dis']
        if not metric in metric_func_name:
            raise ValueError('metric function is not defined.')

        if metric == 'l2_dis':
            def metric_func(u, v):
                return self.calculate_gsc_distance_1(u, v)
        elif metric == 'chi_dis':
            def metric_func(u, v):
                return self.calculate_gsc_distance_2(u, v)
        else:
            metric_func = metric

        mat_Q_Si = np.zeros((num_u, num_si))
        for i in range(num_si):
            index = self.shape_dict.keys()[i]
            sc_i = self.get_shape_gsc_mat(index)
            mat_dis_q_i= spa_dist.cdist(mat_Gi, sc_i, metric_func)
            mat_Q_Si[:, i] = np.amin(mat_dis_q_i, axis=1)

        arr_norm_factor =  np.mean(mat_Q_Si, axis=1)
        for u in range(num_u):
            mat_Q_Si[u, :] = mat_Q_Si[u, :] / arr_norm_factor[u]

        d_Q_Si = np.mean(mat_Q_Si, axis=0)

        voting_dict = {}
        for i in range(num_si):
            voting_dict[self.shape_dict.keys()[i]] = d_Q_Si[i]
            #print self.shape_dict.keys()[i], d_Q_Si[i]
        #[(self.shape_dict.keys()[i], d_Q_Si[i]) for i in range(num_si)]
        #for x, voting in voting_dict.items():print x, '\t', voting
        return voting_dict

    def get_voting_result(self, shape, threshold, metric='l2_dis'):
        Gi = np.array(shape.rsc.values())
        num_rsc = len(shape.rsc)
        voting_dict =self.calculate_voting_distance(mat_Gi=Gi, num_u=num_rsc, num_si=self.shape_size,
                                       num_gsc=self.shape_gsc_num, metric=metric)
        voting_sorted = [v for v in sorted(voting_dict.items(), lambda x, y: cmp(x[1], y[1]))]
        for label, voting in voting_sorted:
            print label, '\t', voting

        voting_result = [label for label, voting in voting_dict.items() if voting <= threshold]
        if not voting_result:
            raise ValueError('threshold is too small!')
        return voting_result



    get_shape_gsc_mat = lambda self, index: np.array(self.shape_dict[index].gsc.values())
