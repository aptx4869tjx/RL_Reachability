import copy
import math
import time

from scipy.optimize import minimize, Bounds

from abstract.tora.tora_abs import *

from verify.divide_tool import str_to_list, list_to_str, max_min_clip, combine_bound_list, contain, split


class TORAEnv1:
    def __init__(self, divide_tool, agent):
        self.divide_tool = divide_tool
        self.network = agent.network
        self.agent = agent
        self.time_seg = 0
        self.time_agg = 0
        self.time_op = 0

    def timer_reset(self):
        self.time_seg = 0
        self.time_agg = 0
        self.time_op = 0

    def get_next_states(self, current):
        t0 = time.time()
        cs = None
        try:
            if isinstance(current, str):
                cs = current
                current = str_to_list(current)
        except:
            print(current)
            exit(0)

        # 0.32 >= x1_new >= -0.32 and 0.32 >= x2_new >= -0.32 and 0.32 >= x3_new >= -0.32
        if current[0] >= -0.1 and current[4] <= 0.2 and current[1] >= -0.9 and current[5] <= -0.6:
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
        cur_list = self.agent.divide_tool.intersection(current)
        original = None
        for cu in cur_list:
            cu = str_to_list(cu)
            if cu[0] <= current[0] and cu[1] <= current[1] and cu[2] <= current[2] and cu[3] <= current[3] and cu[4] >= \
                    current[4] and cu[5] >= current[5] and cu[6] >= current[6] and cu[7] >= current[7]:
                original = cu
                break
        if original == None:
            print('bug')

        s0 = torch.tensor(torch.Tensor(original), dtype=torch.float).unsqueeze(0)
        act = self.network(s0).squeeze(0).detach().numpy()

        next_bounds = self.next_abstract_domain(current, act)

        return next_bounds

    def next_abstract_domain(self, current, act):
        x1l, x2l, x3l, x4l, x1u, x2u, x3u, x4u = tuple(current)

        u1 = np.clip(act, -10, 10)[0]

        t = 0.02

        # x1_new = x1 + x2 * t
        # x2_new = x2 + (-x1 + 0.1 * math.sin(x3)) * t
        # x3_new = x3 + x4 * t
        # x4_new = x4 + u1 * t

        nx1l = x1l + x2l * t
        nx1u = x1u + x2u * t

        xc = [(x1l + x1u) / 2, (x2l + x2u) / 2, (x3l + x3u) / 2]
        bounds = Bounds([x1l, x2l, x3l], [x1u, x2u, x3u])
        # vel_left = minimize(self.x2_minimum, xc, method='SLSQP', bounds=bounds)
        # nx2l = self.x2_minimum(vel_left.x)
        #
        # vel_right = minimize(self.x2_maximum, xc, method='SLSQP', bounds=bounds)
        # nx2u = -self.x2_maximum(vel_right.x)

        nx2l = x2l + (-x1u + 0.1 * math.sin(x3l)) * t
        nx2u = x2u + (-x1l + 0.1 * math.sin(x3u)) * t

        nx3l = x3l + x4l * t
        nx3u = x3u + x4u * t
        nx4l = x4l + u1 * t
        nx4u = x4u + u1 * t

        nx1l = np.clip(nx1l, -2, 2)
        nx1u = np.clip(nx1u, -2, 2)
        nx2l = np.clip(nx2l, -2, 2)
        nx2u = np.clip(nx2u, -2, 2)
        nx3l = np.clip(nx3l, -math.pi / 2, math.pi / 2)
        nx3u = np.clip(nx3u, -math.pi / 2, math.pi / 2)
        nx4l = np.clip(nx4l, -2, 2)
        nx4u = np.clip(nx4u, -2, 2)

        return [nx1l, nx2l, nx3l, nx4l, nx1u, nx2u, nx3u, nx4u]

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
        return u

    def x2_minimum(self, x):
        return x[1] + (-x[0] + 0.1 * math.sin(x[2])) * 0.02

    def x2_maximum(self, x):
        return -self.x2_minimum(x)


# 判断两个bound能否合并
def near_bound(current, target):
    standard = [0.001, 0.001, 0.001, 0.001]  # relu3_20 relu4_100 tanh4_100
    standard = [0.0006, 0.0006, 0.001, 0.001]  # tanh3_20
    # standard = [0.0001, 0.0001, 0.0001, 0.0001]
    # standard = [0.0005, 0.0005, 0.001, 0.001]
    # standard = [0.0004, 0.0004, 0.001, 0.001]
    # standard = [0.001, 0.001, 0.001, 0.001]
    # standard = [0.0004, 0.0004, 0.001, 0.001]
    dim = len(current)
    half_dim = int(dim / 2)
    counter = 0
    record_dim = None
    for i in range(half_dim):
        if abs(current[i] - target[i]) > standard[i] or abs(current[i + half_dim] - target[i + half_dim]) > standard[i]:
            counter += 1
            record_dim = i
    if counter <= 0 or contain(current, target):
        return True
    elif counter == 1:
        if current[record_dim] - target[record_dim + half_dim] <= standard[record_dim] or target[record_dim] - current[
            record_dim + half_dim] <= standard[record_dim]:
            return True
        elif (target[record_dim] < current[record_dim] < target[record_dim + half_dim]) or (
                current[record_dim] < target[record_dim] < current[record_dim + half_dim]):
            return True
        else:
            return False
    else:
        return False
