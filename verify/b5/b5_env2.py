import copy
import time

import torch
from scipy.optimize import minimize, Bounds
import math
import numpy as np

from verify.divide_tool import str_to_list, max_min_clip, combine_bound_list, contain


class B5_Env2():

    def __init__(self, divide_tool, network):
        self.action = 0.0

        self.divide_tool = divide_tool
        self.network = network

        self.time_seg = 0
        self.time_agg = 0
        self.time_op = 0

    def timer_reset(self):
        self.time_seg = 0
        self.time_agg = 0
        self.time_op = 0

    def get_next_bound_list(self, bound_list):
        res_list = []
        cnt = 0
        for bound in bound_list:
            next_bound_list, counter = self.get_next_states(bound)
            cnt += counter
            for next_bound in next_bound_list:
                res_list.append(next_bound)
        t1 = time.time()
        u = combine_bound_list(res_list, near_bound)
        t2 = time.time()
        self.time_agg = self.time_agg + t2 - t1
        # print(cnt)
        return u

    # Method must be implemented by users
    def get_next_states(self, current):
        cs = None
        try:
            if isinstance(current, str):
                cs = current
                current = str_to_list(current)
            else:
                cs = copy.deepcopy(current)
        except:
            print(current)
            exit(0)
        if current[0] > -0.4 and current[3] < -0.28 and current[1] > 0.05 and current[4] < 0.22:
            return [], 0

        # segmentation
        t0 = time.time()
        target_list = self.divide_tool.intersection(current)
        # print(len(target_list))
        tl = []
        for ele in target_list:
            s = str_to_list(ele)
            s = max_min_clip(current, s)
            tl.append(s)
        t1 = time.time()
        self.time_seg = self.time_seg + t1 - t0
        # over-approximation
        b_list = []
        for t in tl:
            b = self.gn(t)
            b_list.append(b)
        t2 = time.time()
        self.time_op = self.time_op + t2 - t1
        # aggregation
        res = combine_bound_list(b_list, near_bound)
        t3 = time.time()
        self.time_agg = self.time_agg + t3 - t2
        return res, len(b_list)

    def gn(self, current):
        cur_list = self.divide_tool.intersection(current)
        original = None
        for cu in cur_list:
            cu = str_to_list(cu)
            if cu[0] <= current[0] and cu[1] <= current[1] and cu[2] <= current[2] and cu[3] >= current[3] and cu[4] >= \
                    current[4] and cu[5] >= current[5]:
                original = cu
                break
        if original == None:
            print('bug')

        s0 = torch.tensor(original, dtype=torch.float).unsqueeze(0)
        action = self.network(s0).squeeze(0).detach().numpy()
        t = 0.05
        # self.action = np.clip(action[0], -1, 1)
        offset = 0
        scala = 1
        self.action = (action[0] - offset) * scala
        x1l = current[0]
        x2l = current[1]
        x3l = current[2]
        x1r = current[3]
        x2r = current[4]
        x3r = current[5]
        # x1_new = x1 + (x1 * x1 * x1 - x2) * t
        # x2_new = x2 + x3 * t
        # x3_new = x3 + u * t
        x1_l = x1l + (x1l * x1l * x1l - x2r) * t
        x1_r = x1r + (x1r * x1r * x1r - x2l) * t
        x2_l = x2l + x3l * t
        x2_r = x2r + x3r * t
        x3_l = x3l + self.action * t
        x3_r = x3r + self.action * t

        x1_l = np.clip(x1_l, -2, 2)
        x1_r = np.clip(x1_r, -2, 2)
        x2_l = np.clip(x2_l, -2, 2)
        x2_r = np.clip(x2_r, -2, 2)
        x3_l = np.clip(x3_l, -5, 5)
        x3_r = np.clip(x3_r, -5, 5)

        next_bounds = [x1_l, x2_l, x3_l, x1_r, x2_r, x3_r]

        return next_bounds


def near_bound(current, target):
    # standard = [0.001, 0.000001]
    # standard = [0.000001, 0.000001]
    standard = [0.001, 0.001, 0.001] # tanh 3_100 4_200 relu 3_100 relu 4_200
    # standard = [0.00001, 0.00001, 0.00001]
    # standard = [0.00001, 0.000001, 0.000001]
    # standard = [0.1, 0.1, 0.1]

    dim = len(current)
    half_dim = int(dim / 2)
    counter = 0
    record_dim = None
    for i in range(half_dim):
        if abs(current[i] - target[i]) > standard[i] or abs(current[i + half_dim] - target[i + half_dim]) > \
                standard[i]:
            counter += 1
            record_dim = i
    if counter <= 0 or contain(current, target):
        return True
    elif counter == 1:
        if current[record_dim] - target[record_dim + half_dim] <= standard[record_dim] or target[record_dim] - \
                current[
                    record_dim + half_dim] <= standard[record_dim]:
            return True
        elif (target[record_dim] < current[record_dim] < target[record_dim + half_dim]) or (
                current[record_dim] < target[record_dim] < current[record_dim + half_dim]):
            return True
        else:
            return False
    else:
        return False
