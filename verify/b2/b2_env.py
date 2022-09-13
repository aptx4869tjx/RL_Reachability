import copy
import time

import torch

import math
import numpy as np

from verify.divide_tool import str_to_list, combine_bound_list, contain, max_min_clip


class B2_Env2():

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

    def get_abstract_state(self, s):
        return self.divide_tool.get_abstract_state(s)

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
        if current[0] >= -0.3 and current[2] <= 0.1 and current[1] >= -0.35 and current[3] <= 0.5:
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

        # target_list = self.divide_tool.intersection(current)
        # tl = []
        # for ele in target_list:
        #     s = str_to_list(ele)
        #     s = max_min_clip(current, s)
        #     tl.append(s)
        # b_list = []
        # for t in tl:
        #     b = self.gn(t)
        #     b_list.append(b)
        #
        # res = combine_bound_list(b_list, near_bound)
        # return res, len(b_list)

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
        # self.action = np.clip(action[0], -self.th, self.th)
        self.action = (action[0] - 0) * 2
        pl = current[0]
        vl = current[1]
        pr = current[2]
        vr = current[3]
        # pl,vl,pr,vr
        # v_new = v + (u * v * v - p) * t
        p_l = pl + (vl - pr * pr * pr) * t
        p_r = pr + (vr - pl * pl * pl) * t
        v_l = vl + self.action * t
        v_r = vr + self.action * t
        p_l = np.clip(p_l, -2, 2)
        p_r = np.clip(p_r, -2, 2)
        v_l = np.clip(v_l, -2, 2)
        v_r = np.clip(v_r, -2, 2)

        next_bounds = [p_l, v_l, p_r, v_r]
        # next_states = self.divide_tool.intersection(next_bounds)
        return next_bounds


def near_bound(current, target):
    # standard = [0.001, 0.000001]
    # standard = [0.000001, 0.000001]
    standard = [0.001, 0.0001]  # relu 3_100 relu 2_20
    # standard = [0.001,0.001]

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
