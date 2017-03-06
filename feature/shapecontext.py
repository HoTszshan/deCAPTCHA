from matplotlib import pyplot as plt
from skimage import feature as ski_feature
from scipy.interpolate import griddata
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance as spa_dist
import scipy.stats as stats
from lib import imgio as ImgIO
from lib import process
import numpy as np

import random
import math
import copy
import time


max_gray_level = 2 ** 8 - 1


class ShapeContext:
    def __init__(self, image, sigma=3.0, sample_num=100):
        self.sigma = sigma
        theta_image = self.normalize_image(image)
        shape_image = self.get_contours()

        sample_points, self.norm_sample_points, self.theta = self.__sampling_edges_3(shape_image, theta_image,
                                                number=sample_num, nearest_coor=self.__get_nearest_coor)
        mat_dist, mat_angle = self.calculate_mat_distIndex_angleIndex(self.norm_sample_points)
        self.sc_dict = self.calculate_shapecontext(self.norm_sample_points, mat_dist, mat_angle)

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
        theta = np.arctan2(dy, dx) + np.pi / 2.0
        return theta

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
        """
        try:
            self.mean_dist
        except AttributeError:
            self.mean_dist = np.mean(mat_dis)
        """
        self.mean_dist = np.mean(mat_dis)

        mat_dis = mat_dis / self.mean_dist
        mat_dis[mat_dis == 0] = 2 ** logr
        mat_dis_logr = np.floor(np.log2(mat_dis)) + (logr - 1)
        #mat_dis_logr = np.log2(mat_dis).astype(np.int) + (logr - 2)
        mat_dis_logr[mat_dis_logr < 0] = 0
        mat_dis_logr[np.all([mat_dis_logr >= logr, mat_dis_logr < 2 * logr -1], axis=0)] = logr - 1

        #print mat_dis_logr[:, 0].tolist()
        return mat_dis_logr, mat_angle_index

    def calculate_shapecontext(self, norm_point_list, mat_disI, mat_angI, logr=5, theta=30):
        sc_dict = {}
        samp_num = len(norm_point_list)
        ang_num = 360 / theta
        bins_num = logr * ang_num
        for i in range(samp_num):
            tmp_sc = np.zeros(bins_num)
            for r in range(logr):
                r_indices = np.where(mat_disI[:, i]==r)[0]
                for j in r_indices:
                    if not i == j:
                        angle_index = int(mat_angI[j, i])
                        tmp_sc[ang_num * r + angle_index] += 1
            tmp_sc = tmp_sc / tmp_sc.sum()
            sc_dict[i] = tmp_sc
        return sc_dict

    def display_norm_points(self, norm_points, symbol='rx'):
        new_img = np.ones((self.height, self.width, 3)) * 255
        plt.imshow(np.uint8(new_img))
        col = [self.get_y_coordinate(point[1]) for point in norm_points]
        row = [self.get_x_coordinate(point[0]) for point in norm_points]
        plt.plot(col, row, symbol)
        plt.axis('off')
        plt.axis('equal')
        plt.show()

    def print_sc(self, tmp_sc, logr=5, ang_num=12):
        for r_ in range(logr):
            for t_ in range(ang_num):
                print ("%.5f" % (tmp_sc[ang_num*r_ + t_])), '\t',
            print ''
        print tmp_sc.sum()

    def __sampling_edges_3(self, shape_img, theta_img, number, nearest_coor, patch=3):
        ratio = float(number) / float(shape_img.sum())
        point_list = []
        if ratio < 1:
            for row in range(0, self.height, patch):
                for col in range(0, self.width, patch):
                    tmp_list = [[row+j, col+i] for j in range(patch) for i in range(patch)
                            if 0 <= row+j < self.height and 0 <= col+i < self.width and shape_img[row+j, col+i]]
                    sample_num = int(math.ceil(len(tmp_list) * ratio))
                    while len(tmp_list) > sample_num:
                        #tmp_list.remove(tmp_list[random.randint(0, len(tmp_list)-1)])
                        random.shuffle(tmp_list)
                        dis_mat = spa_dist.squareform(spa_dist.pdist(tmp_list)) + np.eye(len(tmp_list)) * len(tmp_list) ** 2
                        tmp_list.remove(tmp_list[nearest_coor(dis_mat)])
                    point_list.extend(tmp_list)
            while len(point_list) > number:
                num = len(point_list)
                random.shuffle(point_list)
                dis_mat = spa_dist.squareform(spa_dist.pdist(point_list))
                dis_mat += np.eye(num) * num ** 2
                point_list.remove(point_list[nearest_coor(dis_mat)])
        else:
            point_list.extend([[row, col] for row in range(self.height)
                          for col in range(self.width) if shape_img[row, col]])
            raise ValueError('This shape does not have enough samples')

        normalized_point_list = [[self.get_x_normalized(x), self.get_y_normalized(y)] for x, y in point_list]
        theta_list = [theta_img[x][y] for x, y in point_list]
        return np.array(point_list), np.array(normalized_point_list), np.array(theta_list)

    def __sampling_edges_2(self, shape_img, theta_img, number, nearest_coor):
        # Jitendra: unit length space sampling (for source matlab code)
        point_list = [(i, j) for j in range(self.width) for i in range(self.height) if shape_img[i][j]]
        random.shuffle(point_list)
        while len(point_list) > number:
            num = len(point_list)
            dis_mat = spa_dist.squareform(spa_dist.pdist(point_list))
            dis_mat += np.eye(num) * num ** 2
            point_list.remove(point_list[nearest_coor(dis_mat)])
        normalized_point_list = [[self.get_x_normalized(x), self.get_y_normalized(y)] for x, y in point_list]
        theta_list = [theta_img[x][y] for x, y in point_list]
        return np.array(point_list), np.array(normalized_point_list), np.array(theta_list)

    def __sampling_edges_1(self, shape_img, theta_img, number, patch=3):
        ratio = float(number) / float(shape_img.sum())
        point_list = []
        if ratio < 1:
            for row in range(0, self.height, patch):
                for col in range(0, self.width, patch):
                    tmp_list = [(row+j, col+i) for j in range(patch) for i in range(patch)
                            if 0 <= row+j < self.height and 0 <= col+i < self.width and shape_img[row+j, col+i]]
                    sample_num = int(math.ceil(len(tmp_list) * ratio))
                    while len(tmp_list) > sample_num:
                        tmp_list.remove(tmp_list[random.randint(0,len(tmp_list)-1)])
                    point_list.extend(tmp_list)
            bigger_patch = patch
            while len(point_list) > number:
                tmp_ratio = float(number) / len(point_list)
                update_point_list = []
                bigger_patch = bigger_patch * patch if bigger_patch < self.width else bigger_patch
                for row in range(0, self.height, bigger_patch):
                    for col in range(0, self.width, bigger_patch):
                        tmp_list = [(i, j) for i, j in point_list if row <= i < row+bigger_patch and col <= j < col+bigger_patch]
                        new_sample_num = int(math.ceil(len(tmp_list) * tmp_ratio))
                        while len(tmp_list) > new_sample_num:
                            tmp_list.remove(tmp_list[random.randint(0, len(tmp_list)-1)])
                        update_point_list.extend(tmp_list)
                point_list = update_point_list
        else:
            point_list.extend([(row, col) for row in range(self.height)
                          for col in range(self.width) if shape_img[row, col]])
            raise ValueError('shape #2 does not have enough samples')

        normalized_point_list = [[self.get_x_normalized(x), self.get_y_normalized(y)] for x, y in point_list]
        theta_list = [theta_img[x][y] for x, y in point_list]
        return np.array(point_list), np.array(normalized_point_list), np.array(theta_list)

    def display_sampling(self, sigma=3.0, patch=3, number=100):
        shape_image = ski_feature.canny(self.image, sigma)
        get_min_dis_coor1 = lambda matrix: np.argmin(np.min(matrix, 1))
        p1, norm_1 = self.__sampling_edges_1(shape_image, number, patch)
        p2, norm_2 = self.__sampling_edges_2(shape_image, number, get_min_dis_coor1)
        p3, norm_3 = self.__sampling_edges_3(shape_image, number, get_min_dis_coor1, patch)
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
        plt.plot(y3, x3, "gx")
        plt.axis('equal')
        plt.axis('off')
        plt.show()

    def get_contours(self):
        return ski_feature.canny(self.image, self.sigma)

    def update_sampling_points(self, sample_points):
        self.norm_sample_points = sample_points
        mat_dist, mat_angle = self.calculate_mat_distIndex_angleIndex(sample_points)
        self.sc_dict = self.calculate_shapecontext(sample_points, mat_dist, mat_angle)
        #self.display_norm_points(self.norm_sample_points)
        #self.print_sc(self.sc_dict[0])

    def update_sampling_points_theta(self, sample_points, update_theta):
        self.norm_sample_points = sample_points
        self.theta = update_theta #+ np.pi / 2.0
        mat_dist, mat_angle = self.calculate_mat_distIndex_angleIndex(sample_points)
        self.sc_dict = self.calculate_shapecontext(sample_points, mat_dist, mat_angle)


    get_x_normalized = lambda self, x: float(x) / float(self.height)

    get_y_normalized = lambda self, y: float(y) / float(self.width)

    get_x_coordinate = lambda self, x: int(x * self.height)

    get_y_coordinate = lambda self, y: int(y * self.width)

    __get_nearest_coor = lambda self, matrix: np.argmin(np.min(matrix, 1))


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
        matching_list = assignment(mat)
        matching_list = [list(pair) for pair in matching_list]
        matching_list = sorted(matching_list, key=lambda x: x[0])
        return np.array(matching_list)


def hungarian_assignment(cost_matix):
    row_ind, col_ind = linear_sum_assignment(cost_matix)
    result = np.hstack((np.mat(row_ind).T, np.mat(col_ind).T))
    return result


class TPS_Morpher:

    def __init__(self, points_o, points_t, l_0=0):
        self.l_0 = l_0
        self.points_O = points_o
        self.points_T = points_t
        self.fx, self.fy = self.calculate_TPS_weights(self.points_O, self.points_T)

    def calculate_TPS_weights(self, points_A, points_B):
        mat_A = spa_dist.squareform(spa_dist.pdist(points_A, metric='euclidean'))
        """
        try:
            self.alpha
        except AttributeError:
        """
        self.alpha = mat_A.mean()

        mat_I = np.mat(np.eye(len(points_A))) * self.alpha ** 2 * self.l_0
        mat_K = mat_A * mat_A
        mat_K[mat_K == 0] = 1
        self.mat_K = np.mat(mat_K * np.log(mat_K)) + mat_I
        mat_P = np.mat(np.ones((len(points_A), 3)))
        mat_P[:, 1:3] = points_A
        mat_O = np.mat(np.zeros((3, 3)))
        mat_L = np.mat(np.vstack((np.hstack((self.mat_K, mat_P)), np.hstack((mat_P.T, mat_O)))))

        vector_bx = np.concatenate([points_B[:, 0], np.zeros(3)])
        vector_by = np.concatenate([points_B[:, 1], np.zeros(3)])
        fx = np.linalg.solve(mat_L, vector_bx)
        fy = np.linalg.solve(mat_L, vector_by)

        return (fx[-3], fx[-2], fx[-1], fx[:-3]), (fy[-3], fy[-2], fy[-1], fy[:-3])

    def __interpolate_z_value(self, a1, a2, a3, w, points_A, new_point):
        dist = spa_dist.cdist(points_A, np.array([new_point]))
        r = dist * dist
        r[r == 0] = 1
        r = r * np.log(r)
        kernel_values = w * r.T[0]
        return a1 + a2 * new_point[0] + a3 * new_point[1] + sum(kernel_values)

    def __calculate_point_transform_coordinate(self, new_point):
        new_x = self.__interpolate_z_value(self.fx[0], self.fx[1], self.fx[2], self.fx[3], self.points_O, new_point)
        new_y = self.__interpolate_z_value(self.fy[0], self.fy[1], self.fy[2], self.fy[3], self.points_O, new_point)
        return np.array([new_x, new_y])

    def get_transform_coordinates(self, points):
        tmp = np.array(map(self.__calculate_point_transform_coordinate, points))
        #tmp = np.array([self.__calculate_point_transform_coordinate(p) for p in points])
        return tmp

    def get_transform_coordinates_all(self, points):
        dist = spa_dist.cdist(self.points_O, points)
        r = dist * dist
        r[r == 0] = 1
        #U = dist * np.log(dist + np.finfo(float).eps)
        U = r * np.log(r)
        cx = np.array([[self.fx[0], self.fx[1], self.fx[2]]])
        fx_aff = np.dot(cx, np.vstack((np.ones((1, len(points))), points.T)))
        fx_wrp = np.dot(np.array([self.fx[3]]), U)
        tx = fx_aff + fx_wrp

        cy = np.array([[self.fy[0], self.fy[1], self.fy[2]]])
        fy_aff = np.dot(cy, np.vstack((np.ones((1, len(points))), points.T)))
        fy_wrp = np.dot(np.array([self.fy[3]]), U)
        ty = fy_aff + fy_wrp
        return np.hstack((tx.T, ty.T))

    def get_transform_theta(self, points, theta, height, width):
        # norm = np.array([1.0/float(height), 1.0/float(width)])
        tan = np.array([np.cos(theta), np.sin(theta)]).T # * norm
        #points_tan = np.array([self.__calculate_point_transform_coordinate(p) for p in (points + tan)])
        points_tan = self.get_transform_coordinates_all(points + tan)
        tan_diff = points_tan - points
        return np.arctan2(tan_diff[:, 1], tan_diff[:, 0])

    def calculate_bending_energy(self):
        mat_w = np.vstack((self.fx[3], self.fy[3]))
        Q = (mat_w * self.mat_K * mat_w.T)
        #print Q.mean()
        return Q.diagonal().mean()

    def define_target_points(self, points_t):
        self.points_T = points_t
        self.fx, self.fy = self.calculate_TPS_weights(self.points_O, self.points_T)

    def define_origin_points(self, points_o):
        self.points_O = points_o
        self.fx, self.fy = self.calculate_TPS_weights(self.points_O, self.points_T)


class ShapeContextMatcher:

    def __init__(self, sc_p, sc_q, dummy_frac=0.25, type='sc'):
        self.p = sc_p
        self.q = sc_q
        self.type = type
        self.dummy_ratio = dummy_frac

    def get_initial_matching_pairs(self):
        cost_matrix = self.__calculate_cost_matrix(self.p.sc_dict, self.q.sc_dict, self.p.theta, self.q.theta, self.type)
        cost_matrix, cost_mean = self.__normalize_matrix(cost_matrix)
        matching_pairs = hungarian_assignment(cost_matrix)#hungarian_matching(self.cost_matrix)
        return matching_pairs

    def __calculate_cost_matrix(self, p_sc, q_sc, p_theta, q_theta, type='sc', ori_weight=0.5,
                                polarity_flag=True, tangent_flag=False):
        type_name = ['sc', 'gsc_1', 'gsc_2']
        if not type in type_name:
            raise ValueError('the type is not defined.')

        if type == 'gsc_2':
            def point_cost_func(u, v):
                return self.__get_matching_point_gsc_cost(u, v)
        elif type == 'sc' or type == 'gsc_1':
            def point_cost_func(u, v):
                return self.__get_matching_point_sc_cost(u, v)
        else:
            def point_cost_func(u, v):
                return self.__get_matching_point_sc_cost(u, v)

        if len(self.p.sc_dict) == len(self.q.sc_dict):
            cost_mat_shape = spa_dist.cdist(np.array(p_sc.values()), np.array(q_sc.values()), point_cost_func)
            p_t = p_theta.reshape((len(p_theta), 1))
            q_t = q_theta.reshape((1, len(q_theta)))
            theta_diff = np.repeat(p_t, len(p_theta), axis=1) - np.repeat(q_t, len(q_theta), axis=0)
            if polarity_flag: # use edge polarity
                cost_mat_theta = 0.5 * (1 - np.cos(theta_diff))
            else: # ignore edge polarity
                cost_mat_theta = 0.5 * (1 - np.cos(2 * theta_diff))
            if tangent_flag:
                mat_cost = cost_mat_shape * (1 - ori_weight) + cost_mat_theta * ori_weight
            else:
                mat_cost = cost_mat_shape
            return mat_cost
        else:
            raise ValueError("Calculate Cost Matrix Error!")

    def __normalize_matrix(self, matrix):
        return matrix / matrix.mean(), matrix.mean()

    def __get_matching_point_sc_cost(self, sc1, sc2):
        if len(sc1) == len(sc2):
            denominator = sc1 + sc2
            denominator[denominator == 0.0] = 1.0
            diff_cost = ((sc1 - sc2)**2) / denominator
            # diff_cost = [i for i in diff_cost if not i == 0]
            return 0.5 * sum(diff_cost)
        else:
            raise ValueError("Matching Points Eorror: the length of sc_1 is %d, the length of sc_2 is %d"
                             % (len(sc1), len(sc2)))

    def __get_matching_point_gsc_cost(self, sc1, sc2):
        return spa_dist.pdist([sc1, sc2])[0]

    def calculate_shape_distance(self, l_0=1, iter_num=5, win_size=5, ac=1.6, sc=1.0, be=0.3, mode='easy'):
        mode_name = ['easy', 'tang', 'ndum']
        if not mode in mode_name:
            raise ValueError('The model is not defind')

        if mode == 'easy':
            def shape_distance(u, v, w):
                return self.iterate_easy_TPS_model(u, v, w)
        elif mode == 'tang':
            def shape_distance(u, v, w):
                return self.iterate_TPS_model(u, v, w)
        elif mode == 'ndum':
            def shape_distance(u, v, w):
                return self.iterate_TPS_model_without_outlier(u, v, w)
        else:
            def shape_distance(u, v, w):
                return mode_name

        d_ac, d_sc, d_be = shape_distance(l_0, iter_num, win_size)
        total_cost = d_ac * ac + d_sc * sc + d_be * be
        #
        # TODO:: Out put wrapping & calculate result..
        #
        # print ("Appearance Cost: %.8f,\t Shape Context distance: %.8f,\t Bending Energy: %.8f\t" % (d_ac, d_sc, d_be))
        # print ("Total cost:\t d_ac * %.2f = %.8f,\t d_sc * %.2f = %.6f,\t d_be * %.2f = %.6f,\t total_cost = %.8f"
        #       % (ac, d_ac*ac, sc, d_sc*sc, be, d_be*be, total_cost))
        """
        time1 = time.time()
        d_ac, d_sc, d_be = self.iterate_easy_TPS_model(l_0, iter_num, win_size)
        print 'EASY:', time.time() - time1
        total_cost = d_ac * ac + d_sc * sc + d_be * be
        print ("Appearance Cost: %.8f,\t Shape Context distance: %.8f,\t Bending Energy: %.8f\t" % (d_ac, d_sc, d_be))
        print ("Total cost:\t d_ac * %.2f = %.8f,\t d_sc * %.2f = %.6f,\t d_be * %.2f = %.6f,\t total_cost = %.8f"
               % (ac, d_ac*ac, sc, d_sc*sc, be, d_be*be, total_cost))

        time2 = time.time()
        d_ac, d_sc, d_be = self.iterate_TPS_model(l_0, iter_num, win_size)
        print 'Tang:', time.time() - time2
        total_cost = d_ac * ac + d_sc * sc + d_be * be
        print ("Appearance Cost: %.8f,\t Shape Context distance: %.8f,\t Bending Energy: %.8f\t" % (d_ac, d_sc, d_be))
        print ("Total cost:\t d_ac * %.2f = %.8f,\t d_sc * %.2f = %.6f,\t d_be * %.2f = %.6f,\t total_cost = %.8f"
               % (ac, d_ac*ac, sc, d_sc*sc, be, d_be*be, total_cost))

        time3 = time.time()
        d_ac, d_sc, d_be = self.iterate_TPS_model_without_outlier(l_0, iter_num, win_size)
        print 'Ndum:', time.time() - time3
        total_cost = d_ac * ac + d_sc * sc + d_be * be
        print ("Appearance Cost: %.8f,\t Shape Context distance: %.8f,\t Bending Energy: %.8f\t" % (d_ac, d_sc, d_be))
        print ("Total cost:\t d_ac * %.2f = %.8f,\t d_sc * %.2f = %.6f,\t d_be * %.2f = %.6f,\t total_cost = %.8f"
               % (ac, d_ac*ac, sc, d_sc*sc, be, d_be*be, total_cost))
        """
        return total_cost

    def __gaussker_points_result(self, num, gaussfunc, win_size, image, norm_points):
            height, width = image.shape
            points = norm_points * np.array([height, width])
            half_win = win_size / 2
            win_list = np.zeros((num, win_size**2))

            for i in range(num):
                row = int(round(points[i, 0]))
                col = int(round(points[i, 1]))
                row = max(half_win, min(height-half_win-1, row))
                col = max(half_win, min(width-half_win-1, col))
                tmp = image[row-half_win:row+half_win+1, col-half_win:col+half_win+1]
                tmp = tmp * gaussfunc
                win_list[i, :] = tmp.reshape(tmp.size)
            #win_list = map(lambda p: self.__single_gaussker_point(image, gaussfunc, height, width,half_win, p), points)
            return win_list

    def calculate_appearance_cost(self, img_wrp, img_tar, update_points, target_points, opti_q, opti_p, window_size, nsmap):
        win_fun = process.get_gaussker(window_size)
        gauss_1 = self.__gaussker_points_result(nsmap, win_fun, window_size, img_wrp, update_points)
        gauss_2 = self.__gaussker_points_result(nsmap, win_fun, window_size, img_tar, target_points)
        ssd_all = spa_dist.cdist(gauss_1, gauss_2)
        cost_1 = 0.0
        cost_2 = 0.0
        for qq in range(nsmap):
            cost_1 += ssd_all[qq, opti_q[qq]]
            cost_2 += ssd_all[opti_p[qq], qq]
        ssd_local = (1/float(nsmap)) * np.array([cost_1, cost_2]).max()
        ssd_local_avg = (1/float(nsmap)) * np.array([cost_1, cost_2]).mean()
        return ssd_local, ssd_local_avg

    def calculate_shape_context_distance(self, new_sc, target_sc):
        new_cost_mat = self.__calculate_cost_matrix(new_sc.sc_dict, target_sc.sc_dict, new_sc.theta, target_sc.theta, self.type)
        new_cost_mat, mean = self.__normalize_matrix(new_cost_mat)

        min_row_list = np.amin(new_cost_mat, axis=1)
        min_col_list = np.amin(new_cost_mat, axis=0)
        min_row_arg = np.argmin(new_cost_mat, axis=1)
        min_col_arg = np.argmin(new_cost_mat, axis=0)
        return min_row_list.mean() + min_col_list.mean(), min_row_arg, min_col_arg

    def calculate_shape_context_distance_by_Mat(self, matrix):
        min_row_list = np.amin(matrix, axis=1)
        min_col_list = np.amin(matrix, axis=0)
        min_row_arg = np.argmin(matrix, axis=1)
        min_col_arg = np.argmin(matrix, axis=0)
        return (min_row_list.mean() + min_col_list.mean()), min_row_arg, min_col_arg

    def __transformation_TPS_model(self, points_o, points_t, l_0, iter_num):
        t_points = points_o
        tps_model = None
        for i in range(iter_num):
            if not tps_model:
                tps_model = TPS_Morpher(points_o, points_t, l_0)
            else:
                tps_model.define_origin_points(t_points)
            t_points = tps_model.get_transform_coordinates(t_points)
        return tps_model, t_points

    # Easy one: 1 hungarian assignment, without tangent angle
    def iterate_easy_TPS_model(self, l_0, iter_num, win_size):
        cost_matrix = self.__calculate_cost_matrix(self.p.sc_dict, self.q.sc_dict,
                                                   self.p.theta, self.q.theta, self.type, tangent_flag=False)
        cost_matrix, cost_mean = self.__normalize_matrix(cost_matrix)
        matching_pairs = hungarian_assignment(cost_matrix)

        points_P = np.array([self.p.norm_sample_points[matching_pairs[i, 0]] for i in range(len(matching_pairs))])
        points_Q = np.array([self.q.norm_sample_points[matching_pairs[i, 1]] for i in range(len(matching_pairs))])
        tps, new_points = self.__transformation_TPS_model(points_Q, points_P, l_0=l_0, iter_num=iter_num)

        new_sc_Q = copy.deepcopy(self.q)
        new_sc_Q.update_sampling_points(new_points)
        # matching_pairs[:, 1] = matching_pairs[:, 0]
        # self.display_update_shapes_matching(new_sc_Q, self.p, matching_pairs)
        shape_context_distance, opti_q_I, opti_p_I = self.calculate_shape_context_distance(new_sc_Q, self.p)

        T_q = self.grayscale_warping(self.q.image, self.q.height, self.q.width, tps)
        ac_local, ac_avg = self.calculate_appearance_cost(img_wrp=T_q, img_tar=self.p.image, update_points=new_points,
                                                          target_points=points_P, opti_q=opti_q_I, opti_p=opti_p_I,
                                                          window_size=win_size, nsmap=len(self.p.norm_sample_points))
        # ssd = self.q.image - T_q
        # ssd_global = ssd.sum()
        #
        """
        # TODO:: Out put wrapping & calculate result..
        """
        # print ("ssd_local: %.6f,\t\t ssd_local_avg: %.6f,\t\t ssd_global: %.6f\t\t" % (ac_local, ac_avg, ssd_global.sum()))

        return ac_avg, shape_context_distance, tps.calculate_bending_energy()

    # Iteration one: update hungarian assignment, with tangent angle
    def iterate_TPS_model(self, l_0, iter_num, win_size):
        sc_tmp_q = copy.deepcopy(self.q)
        if iter_num <= 0: iter_num = 1
        for i in range(iter_num):
            cost_matrix = self.__calculate_cost_matrix(self.p.sc_dict, sc_tmp_q.sc_dict,
                                                       self.p.theta, sc_tmp_q.theta, self.type, tangent_flag=True)
            cost_matrix, cost_mean = self.__normalize_matrix(cost_matrix)
            matching_pairs = hungarian_assignment(cost_matrix)

            points_target = np.array([self.p.norm_sample_points[matching_pairs[i, 0]] for i in range(len(matching_pairs))])
            points_origin = np.array([sc_tmp_q.norm_sample_points[matching_pairs[i, 1]] for i in range(len(matching_pairs))])
            # self.display_update_shapes_matching(sc_tmp_q, self.p, matching=matching_pairs)
            tps = TPS_Morpher(points_o=points_origin, points_t=points_target, l_0=l_0)
            new_points = tps.get_transform_coordinates_all(points_origin)
            new_thetas = tps.get_transform_theta(points_origin, sc_tmp_q.theta, sc_tmp_q.height, sc_tmp_q.width)
            sc_tmp_q.update_sampling_points_theta(sample_points=new_points, update_theta=new_thetas)
        else:
            shape_context_distance, opti_q_I, opti_p_I = self.calculate_shape_context_distance_by_Mat(cost_matrix)
            T_q = self.grayscale_warping(self.q.image, self.q.height, self.q.width, tps)
            ac_local, ac_avg = self.calculate_appearance_cost(img_wrp=T_q, img_tar=self.p.image, update_points=new_points,
                                                          target_points=points_target, opti_q=opti_q_I, opti_p=opti_p_I,
                                                          window_size=win_size, nsmap=len(self.p.norm_sample_points))
            return ac_avg, shape_context_distance, tps.calculate_bending_energy()

    # Iteration without outlier: update hungarian assignment, with tangent angle
    def iterate_TPS_model_without_outlier(self, l_0, iter_num, win_size):
        sc_tmp_q = copy.deepcopy(self.q)
        if iter_num <= 0: iter_num = 1
        for i in range(iter_num):
            cost_matrix = self.__calculate_cost_matrix(self.p.sc_dict, sc_tmp_q.sc_dict,
                                                       self.p.theta, sc_tmp_q.theta, self.type, tangent_flag=True)
            cost_matrix, cost_mean = self.__normalize_matrix(cost_matrix)
            # matching_pairs = hungarian_assignment(cost_matrix)
            matching_pairs = self.__filter_outliers(cost_matrix)

            points_target = np.array([self.p.norm_sample_points[matching_pairs[i, 0]] for i in range(len(matching_pairs))])
            points_origin = np.array([sc_tmp_q.norm_sample_points[matching_pairs[i, 1]] for i in range(len(matching_pairs))])
            # self.display_update_shapes_matching(sc_tmp_q, self.p, matching=matching_pairs)
            tps = TPS_Morpher(points_o=points_origin, points_t=points_target, l_0=l_0)
            new_points = tps.get_transform_coordinates_all(sc_tmp_q.norm_sample_points)
            new_thetas = tps.get_transform_theta(sc_tmp_q.norm_sample_points, sc_tmp_q.theta, sc_tmp_q.height, sc_tmp_q.width)
            sc_tmp_q.update_sampling_points_theta(sample_points=new_points, update_theta=new_thetas)
        else:
            shape_context_distance, opti_q_I, opti_p_I = self.calculate_shape_context_distance_by_Mat(cost_matrix)
            matching_pairs = hungarian_assignment(cost_matrix)
            points_target = np.array([self.p.norm_sample_points[matching_pairs[i, 0]] for i in range(len(matching_pairs))])
            #shape_context_distance, opti_q_I, opti_p_I = self.calculate_shape_context_distance_by_Mat(cost_matrix)
            T_q = self.grayscale_warping(self.q.image, self.q.height, self.q.width, tps)
            ac_local, ac_avg = self.calculate_appearance_cost(img_wrp=T_q, img_tar=self.p.image, update_points=new_points,
                                                          target_points=points_target, opti_q=opti_q_I, opti_p=opti_p_I,
                                                          window_size=win_size, nsmap=len(self.p.norm_sample_points))
            return ac_avg, shape_context_distance, tps.calculate_bending_energy()

    def __filter_outliers(self, mat_cost):
        height, width = mat_cost.shape
        dummy_number = int(np.round(height * self.dummy_ratio))
        mat_cost_2 = np.ones((height+dummy_number, width+dummy_number)) * self.dummy_ratio
        mat_cost_2[:height, :width] = mat_cost
        matching_pairs = hungarian_assignment(mat_cost_2)
        matching_pairs = matching_pairs[:width]

        out_vec_q = np.array(matching_pairs)[:, 1]
        out_vec_q = out_vec_q[:width] >= width
        out_vec_q_index = np.argwhere(out_vec_q)
        out_vec_q_index = out_vec_q_index.reshape(out_vec_q_index.shape[0],)
        matching_pairs = np.delete(matching_pairs, out_vec_q_index, axis=0)
        """
        matching_pairs_2 = np.array(sorted(matching_pairs_1.tolist(), key=lambda x: x[1]))
        out_vec_p = matching_pairs_2[:, 0]
        out_vec_p = out_vec_p[:height] > height
        """
        return matching_pairs

    def grayscale_warping(self, image, height, width, tps):
        x = np.linspace(0, float(height-1)/float(height), height)
        y = np.linspace(0, float(width-1)/float(width), width)
        xv, yv = np.meshgrid(x, y)
        xv, yv = xv.reshape((xv.size, 1)), yv.reshape((yv.size, 1))
        coors = np.hstack((xv, yv))
        coors_wrp = tps.get_transform_coordinates_all(coors)
        img_wrp = griddata(coors, image.reshape(image.size), coors_wrp, method='linear')
        img_wrp = img_wrp.reshape((height, width))
        img_wrp[np.where(np.isnan(img_wrp))] = 0
        return img_wrp

    def display_two_shapes_matching(self, matching_pairs, symbol1='r.', symbol2='g.'):
        new_img = np.ones((self.p.height, self.p.width, 3)) * 255
        plt.imshow(np.uint8(new_img))

        points_P = self.p.norm_sample_points * np.array([self.p.height, self.p.width])
        points_Q = self.q.norm_sample_points * np.array([self.q.height, self.q.width])
        col1, row1 = points_P[:, 1], points_P[:, 0]
        col2, row2 = points_Q[:, 1], points_Q[:, 0]
        plt.plot(col1, row1, symbol1)
        plt.plot(col2, row2, symbol2)

        for pair in matching_pairs:
            origin = pair[0, 0]
            target = pair[0, 1]
            x = [points_P[origin, 1], points_Q[target, 1]]
            y = [points_P[origin, 0], points_Q[target, 0]]
            plt.plot(x[:], y[:], 'b:')
        plt.axis('off')
        plt.axis('equal')

        plt.show()

    def display_update_shapes_matching(self, new_sc, target_sc, matching, symbol1='r.', symbol2='g.', symbol3='b.'):
        new_img = np.ones((self.p.height, self.p.width, 3)) * 255

        plt.subplot(1, 2, 1)
        plt.imshow(np.uint8(new_img))
        points_P = self.p.norm_sample_points * np.array([self.p.height, self.p.width])
        points_Q = self.q.norm_sample_points * np.array([self.q.height, self.q.width])
        col1, row1 = points_P[:, 1], points_P[:, 0]
        col2, row2 = points_Q[:, 1], points_Q[:, 0]
        plt.plot(col1, row1, symbol1)
        plt.plot(col2, row2, symbol2)
        initial_matching = self.get_initial_matching_pairs()
        for pair in initial_matching:
            origin = pair[0, 1]
            target = pair[0, 0]
            x = [points_Q[origin, 1], points_P[target, 1]]
            y = [points_Q[origin, 0], points_P[target, 0]]
            plt.plot(x[:], y[:], 'b:')
        #plt.quiver(col1, row1, np.cos(self.p.theta), np.sin(self.p.theta), alpha=.5)
        #plt.quiver(col2, row2, np.cos(self.q.theta), np.sin(self.q.theta), alpha=.5)
        plt.axis('off')
        plt.axis('equal')


        plt.subplot(1, 2, 2)
        plt.imshow(np.uint8(new_img))
        new_points = new_sc.norm_sample_points * np.array([new_sc.height, new_sc.width])
        target_points = target_sc.norm_sample_points * np.array([target_sc.height, target_sc.width])
        col3, row3 = new_points[:, 1], new_points[:, 0]
        col4, row4 = target_points[:, 1], target_points[:, 0]
        plt.plot(col3, row3, symbol3)
        plt.plot(col4, row4, symbol1)
        for pair in matching:
            target = pair[0, 1]
            new_or = pair[0, 0]
            x = [new_points[target, 1], target_points[new_or, 1]]
            y = [new_points[target, 0], target_points[new_or, 0]]
            plt.plot(x[:], y[:], 'b:')
        #plt.quiver(col3, row3, np.cos(new_sc.theta), np.sin(new_sc.theta), alpha=.5)
        plt.axis('off')
        plt.axis('equal')
        plt.show()

    def display_tps_wrapping_result(self, l_0=1, symbol1='rx', symbol2='gx', symbol3='b.', number=5):
        new_img = np.ones((self.p.height, self.p.width, 3)) * 255
        row_num = number / 3
        matching_pairs = self.get_initial_matching_pairs()

        P = np.array([self.p.norm_sample_points[matching_pairs[i, 0]] for i in range(len(matching_pairs))])
        Q = np.array([self.q.norm_sample_points[matching_pairs[i, 1]] for i in range(len(matching_pairs))])
        points_P = P * np.array([self.p.height, self.p.width])
        points_Q = Q * np.array([self.q.height, self.q.width])
        col1, row1 = points_P[:, 1], points_P[:, 0]
        col2, row2 = points_Q[:, 1], points_Q[:, 0]

        plt.subplot(row_num+1, 3, 1)
        plt.title('origin points')
        plt.imshow(np.uint8(new_img))
        plt.plot(col1, row1, symbol1)
        plt.plot(col2, row2, symbol2)
        for i in range(len(points_P)):
            x = [points_P[i, 1], points_Q[i, 1]]
            y = [points_P[i, 0], points_Q[i, 0]]
            plt.plot(x[:], y[:], 'b:')
        plt.axis('off')
        plt.axis('equal')

        for img_num in range(1, number+1):
            plt.subplot(row_num+1, 3, img_num+1)
            plt.title('iter_num =' + str(img_num))
            plt.imshow(np.uint8(new_img))

            tps, new_points = self.__transformation_TPS_model(P, Q, l_0=l_0, iter_num=img_num)
            points_updated = new_points * np.array([self.q.height, self.q.width])
            ncol, nrow = points_updated[:, 1], points_updated[:, 0]
            plt.plot(col2, row2, symbol2)
            plt.plot(ncol, nrow, symbol3)
            for i in range(len(points_Q)):
                x = [points_updated[i, 1], points_Q[i, 1]]
                y = [points_updated[i, 0], points_Q[i, 0]]
                plt.plot(x[:], y[:], 'b:')
            plt.axis('off')
            plt.axis('equal')

        plt.show()

    """
    def display_tps_processing(self, points_o, points_t, iter_num=5, l_0=1, symbol_o='rx', symbol_t='gx', symbol_pro='b.'):
        new_img = np.ones((2, 2, 3)) * 255
        plt.imshow(np.uint8(new_img))

        col1, row1 = points_o[:, 1], points_o[:, 0]
        col2, row2 = points_t[:, 1], points_t[:, 0]
        plt.plot(col1, row1, symbol_o)
        plt.plot(col2, row2, symbol_t)

        tmp_shape = [points_o]
        for i in range(1, iter_num):
            tps, updated_points = self.transformation_TPS_model(points_o, points_t, l_0=l_0, iter_num=i)
            tmp_shape.append(updated_points)
            col, row = updated_points[:, 1], updated_points[:, 0]
            plt.plot(col, row, symbol_pro)
        tmp_shape.append(points_t)
        for num in range(len(points_o)):
            y = []
            x = []
            for i in range(iter_num+1):
                y.append(tmp_shape[i][num][1])
                x.append(tmp_shape[i][num][0])
            plt.plot(y[:], x[:], 'b:')
        plt.axis('off')
        plt.axis('equal')

        plt.show()
    """


class GeneralizedShapeContext:
    def __init__(self, image, sigma=3.0, sample='gsc', sample_params=100):
        self.sigma = sigma
        self.normalize_image(image)
        shape_image = self.get_contours()
        if sample == 'gsc':
            sample_points, self.norm_sample_points = self.sampling_edge_gsc(shape_image,
                                                        number=sample_params, nearest_coor=self.__get_nearest_coor)
            mat_dist, mat_angle = self.calculate_mat_distIndex_angleIndex(self.norm_sample_points)
            self.gsc = self.calculate_gsc(sample_points, self.norm_sample_points, mat_dist, mat_angle)
            self.rsc = None
        elif sample == 'rsc':
            sample_points, self.norm_sample_points = self.sampling_edge_gsc(shape_image,
                                                        number=100, nearest_coor=self.__get_nearest_coor)
            mat_dist, mat_angle = self.calculate_mat_distIndex_angleIndex(self.norm_sample_points)
            self.gsc = self.calculate_gsc(sample_points, self.norm_sample_points, mat_dist, mat_angle)
            self.r_points, self.rsc = self.calculate_rsc_1(self.gsc, self.norm_sample_points, ratio=sample_params)
        else:
            print "Sampling method is not define!"
            print "Sampling to generate generalized shape context!"
            sample_points, self.norm_sample_points = self.sampling_edge_gsc(shape_image,
                                                        number=100, nearest_coor=self.__get_nearest_coor)
            mat_dist, mat_angle = self.calculate_mat_distIndex_angleIndex(self.norm_sample_points)
            self.gsc = self.calculate_gsc(sample_points, self.norm_sample_points, mat_dist, mat_angle)
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
        self.shape_dict = self.__initialize_shape_dict(label_list, gsc_list)

    def __initialize_shape_dict(self, label_list, gsc_list):
        tmp_dict = {}
        for i in range(len(label_list)):
            tmp_dict[label_list[i]] = gsc_list[i]
        return tmp_dict

    def __calculate_gsc_distance_1(self, gsc_1, gsc_2):
        dist = sum((gsc_1 - gsc_2) ** 2)
        return math.sqrt(dist)

    def __calculate_gsc_distance_2(self, gsc_1, gsc_2):
        numerator = (gsc_1 - gsc_2) ** 2
        denominator = np.fabs(gsc_1) + np.fabs(gsc_2)
        return sum([numerator[i] / denominator[i] if not denominator[i] == 0 else 0 for i in range(len(gsc_1))])

    def calculate_voting_distance(self, mat_Gi, num_u, num_si, num_gsc, metric='l2_dis'):
        metric_func_name = ['l2_dis', 'chi_dis']
        if not metric in metric_func_name:
            raise ValueError('metric function is not defined.')

        if metric == 'l2_dis':
            def metric_func(u, v):
                return self.__calculate_gsc_distance_1(u, v)
        elif metric == 'chi_dis':
            def metric_func(u, v):
                return self.__calculate_gsc_distance_2(u, v)
        else:
            metric_func = metric

        mat_Q_Si = np.zeros((num_u, num_si))
        for i in range(num_si):
            index = self.shape_dict.keys()[i]
            sc_i = self.__get_shape_gsc_mat(index)
            mat_dis_q_i= spa_dist.cdist(mat_Gi, sc_i, metric_func)
            mat_Q_Si[:, i] = np.amin(mat_dis_q_i, axis=1)

        arr_norm_factor = np.mean(mat_Q_Si, axis=1)
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

    def get_voting_result(self, shape, threshold, low_threshold, metric='l2_dis'):
        Gi = np.array(shape.rsc.values())
        num_rsc = len(shape.rsc)
        voting_dict =self.calculate_voting_distance(mat_Gi=Gi, num_u=num_rsc, num_si=self.shape_size,
                                       num_gsc=self.shape_gsc_num, metric=metric)
        voting_sorted = [v for v in sorted(voting_dict.items(), lambda x, y: cmp(x[1], y[1]))]

        #for label, voting in voting_sorted:
        #    print label, '\t', voting
        voting_result = [label[0] for label, voting in voting_sorted if voting <= threshold]
        #voting_result = [label[0] for label, voting in voting_dict.items() if voting <= threshold]
        #"""
        if voting_sorted[0][1] < low_threshold and len(voting_result) > 3:
            if voting_result[0] == voting_result[1]:
                voting_result = voting_result[:2]
            else:
                voting_result = voting_result[:4]

        if not voting_result:
            raise ValueError('threshold is too small!')

        voting_result = voting_result[:8] if len(voting_result) > 8 else voting_result
        voting_result = {}.fromkeys(voting_result).keys()
        voting_result = sorted(voting_result)
        return (voting_result, voting_sorted)


    __get_shape_gsc_mat = lambda self, index: np.array(self.shape_dict[index].gsc.values())
