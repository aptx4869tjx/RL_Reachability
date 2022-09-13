import copy
import time

import torch

from scipy.optimize import minimize, Bounds
import math
import numpy as np

from verify.divide_tool import str_to_list, combine_bound_list, contain, max_min_clip


class B3_Env():

    def __init__(self, divide_tool, network):
        self.action = 0.0
        self.tau = 0.02

        self.divide_tool = divide_tool
        self.network = network
        self.time_seg = 0
        self.time_agg = 0
        self.time_op = 0

    def timer_reset(self):
        self.time_seg = 0
        self.time_agg = 0
        self.time_op = 0


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
        if current[0] >= 0.2 and current[2] <= 0.3 and current[1] >= -0.3 and current[3] <= -0.05:
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

    def gn(self, current):
        cur_list = self.divide_tool.intersection(current)
        original = None
        for cu in cur_list:
            cu = str_to_list(cu)
            if cu[0] <= current[0] and cu[1] <= current[1] and cu[2] >= current[2] and cu[3] >= current[3]:
                original = cu
                break
        if original == None:
            print('bug')

        s0 = torch.tensor(original, dtype=torch.float).unsqueeze(0)
        action = self.network(s0).squeeze(0).detach().numpy()
        t = 0.05
        self.tau = t
        # x1_new = x1 - x1 * (0.1 + (x1 + x2) * (x1 + x2)) * t
        # x2_new = x2 + (u + x1) * (0.1 + (x1 + x2) * (x1 + x2)) * t
        offset = 0
        scala = 1

        # self.action = np.clip(action[0], -self.th, self.th)


        # offset = 0.5
        # scala = 1
        self.action = (action[0] - offset) * scala
        x1l = current[0]
        x2l = current[1]
        x1r = current[2]
        x2r = current[3]
        # pl,vl,pr,vr
        # v_new = v + (u * v * v - p) * t
        bound = Bounds([x1l, x2l], [x1r, x2r])
        x0 = [(x1l + x1r) / 2, (x2l + x2r) / 2]
        x1_l = minimize(self.x1_minimum, x0, method='SLSQP', bounds=bound)
        x1_l = self.x1_minimum(x1_l.x)
        x1_r = minimize(self.x1_maximum, x0, method='SLSQP', bounds=bound)
        x1_r = -self.x1_maximum(x1_r.x)

        x2_l = minimize(self.x2_minimum, x0, method='SLSQP', bounds=bound)
        x2_l = self.x2_minimum(x2_l.x)

        x2_r = minimize(self.x2_maximum, x0, method='SLSQP', bounds=bound)
        x2_r = -self.x2_maximum(x2_r.x)

        x1_l = np.clip(x1_l, -2, 2)
        x1_r = np.clip(x1_r, -2, 2)
        x2_l = np.clip(x2_l, -2, 2)
        x2_r = np.clip(x2_r, -2, 2)

        next_bounds = [x1_l, x2_l, x1_r, x2_r]
        # next_states = self.divide_tool.intersection(next_bounds)
        return next_bounds

    def x1_minimum(self, x):
        return x[0] - x[0] * (0.1 + (x[0] + x[1]) * (x[0] + x[1])) * self.tau

    def x1_maximum(self, x):
        return -self.x1_minimum(x)

    def x2_minimum(self, x):
        return x[1] + (self.action + x[0]) * (0.1 + (x[0] + x[1]) * (x[0] + x[1])) * self.tau

    def x2_maximum(self, x):
        return -self.x2_minimum(x)


def near_bound(current, target):
    # standard = [0.001, 0.000001]
    # standard = [0.000001, 0.000001]

    standard = [0.001, 0.0001] # relu 3_100 tanh 2_20 tanh 3_100
    # standard = [0.0001, 0.0001]
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
