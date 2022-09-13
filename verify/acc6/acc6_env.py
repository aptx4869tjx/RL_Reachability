import copy
import time

from scipy.optimize import minimize

from abstract.qmpc.qmpc_abs import *

from verify.divide_tool import str_to_list, list_to_str, max_min_clip, combine_bound_list, contain


class ACC6Env1:
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
        if current[1] > 22.81 and current[7] < 22.87 and current[4] > 29.88 and current[10] < 30.02:
            return [], 0, []
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
        b_list = []
        res_interval_list = []
        for t in tl:
            b, interval_box_list = self.gn(t)
            b_list.append(b)
            res_interval_list.append(interval_box_list)
        t2 = time.time()
        self.time_op = self.time_op + t2 - t1
        res = combine_bound_list(b_list, near_bound)
        t3 = time.time()
        self.time_agg = self.time_agg + t3 - t2
        return res, len(b_list), res_interval_list

    def get_low_dim_state(self, state):
        s = str_to_list(state)
        res = [s[0], s[2], s[4], s[6]]
        obj_str = ','.join([str(_) for _ in res])
        return obj_str

    def gn(self, current):
        cur_list = self.agent.divide_tool.intersection(current)
        original = None
        for cu in cur_list:
            cu = str_to_list(cu)
            if cu[0] <= current[0] and cu[1] <= current[1] and cu[2] <= current[2] and cu[3] <= current[3] and cu[4] <= \
                    current[4] and cu[5] <= current[5] and cu[6] >= current[6] and cu[7] >= current[7] and cu[8] >= \
                    current[8] and cu[9] >= current[9] and cu[10] >= current[10] and cu[11] >= current[11]:
                original = cu
                break
        if original == None:
            print('bug')

        s0 = torch.tensor(torch.Tensor(original), dtype=torch.float).unsqueeze(0)
        act = self.network(s0).squeeze(0).detach().numpy()

        next_bounds, interval_box_list = self.next_abstract_domain(current, act)

        return next_bounds, interval_box_list

    def next_abstract_domain(self, current, act):
        x1l, x2l, x3l, x4l, x5l, x6l, x1u, x2u, x3u, x4u, x5u, x6u = tuple(current)

        interval_box_list = []
        # interval_box_list=[[x1l, x2l, x3l, x4l, x5l, x6l, x1u, x2u, x3u, x4u, x5u, x6u]]
        u = act[0]
        t = 0.1
        mini_t = 0.01
        cur_t = 0
        # t = 0.1
        # x1_new = x1 + x2 * t
        # x2_new = x2 + x3 * t
        # x3_new = x3 + (2 * u1 - 2 * x3 - x2 * x2 / 10000) * t
        # x4_new = x4 + x5 * t
        # x5_new = x5 + x6 * t
        # x6_new = x6 + (2 * u2 - 2 * x6 - x5 * x5 / 10000) * t
        nx1l, nx2l, nx3l, nx4l, nx5l, nx6l, nx1u, nx2u, nx3u, nx4u, nx5u, nx6u = x1l, x2l, x3l, x4l, x5l, x6l, x1u, x2u, x3u, x4u, x5u, x6u
        while cur_t < t:
            x2_abs_min = min(abs(x2l), abs(x2u))
            x2_abs_max = max(abs(x2l), abs(x2u))
            x5_abs_min = min(abs(x5l), abs(x5u))
            x5_abs_max = max(abs(x5l), abs(x5u))

            nx1l = x1l + x2l * mini_t
            nx1u = x1u + x2u * mini_t
            nx2l = x2l + x3l * mini_t
            nx2u = x2u + x3u * mini_t
            nx3l = x3l + (-4 - 2 * x3l - x2_abs_max * x2_abs_max / 10000) * mini_t
            nx3u = x3u + (-4 - 2 * x3u - x2_abs_min * x2_abs_min / 10000) * mini_t
            nx4l = x4l + x5l * mini_t
            nx4u = x4u + x5u * mini_t
            nx5l = x5l + x6l * mini_t
            nx5u = x5u + x6u * mini_t
            nx6l = x6l + (2 * u - 2 * x6l - x5_abs_max * x5_abs_max / 10000) * mini_t
            nx6u = x6u + (2 * u - 2 * x6u - x5_abs_min * x5_abs_min / 10000) * mini_t
            interval_box_list.append([nx1l, nx2l, nx3l, nx4l, nx5l, nx6l, nx1u, nx2u, nx3u, nx4u, nx5u, nx6u])
            x1l, x2l, x3l, x4l, x5l, x6l, x1u, x2u, x3u, x4u, x5u, x6u = nx1l, nx2l, nx3l, nx4l, nx5l, nx6l, nx1u, nx2u, nx3u, nx4u, nx5u, nx6u
            cur_t = round(cur_t + mini_t, 10)

        # nx1l = np.clip(nx1l, 0, 199)
        # nx1u = np.clip(nx1u, 0, 199)
        # nx2l = np.clip(nx2l, -99, 99)
        # nx2u = np.clip(nx2u, -99, 99)
        # nx3l = np.clip(nx3l, -99, 99)
        # nx3u = np.clip(nx3u, -99, 99)
        # nx4l = np.clip(nx4l, 0, 199)
        # nx4u = np.clip(nx4u, 0, 199)
        # nx5l = np.clip(nx5l, -99, 99)
        # nx5u = np.clip(nx5u, -99, 99)
        # nx6l = np.clip(nx6l, -99, 99)
        # nx6u = np.clip(nx6u, -99, 99)

        return [nx1l, nx2l, nx3l, nx4l, nx5l, nx6l, nx1u, nx2u, nx3u, nx4u, nx5u, nx6u], interval_box_list

    def get_next_bound_list(self, bound_list):
        res_list = []
        bound_interval_list = []
        cnt = 0
        for bound in bound_list:
            next_bound_list, counter, res_interval_list = self.get_next_states(bound)
            cnt += counter
            for next_bound in next_bound_list:
                res_list.append(next_bound)
            for interval_box_list in res_interval_list:
                bound_interval_list.append(interval_box_list)
        t1 = time.time()
        u = combine_bound_list(res_list, near_bound)
        t2 = time.time()
        self.time_agg = self.time_agg + t2 - t1
        # print(cnt)
        return u, bound_interval_list


# 判断两个bound能否合并
def near_bound(current, target):
    standard = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
    # standard = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    # standard = [5, 5, 5, 5, 5, 5]
    # standard = [0.0001, 0.01, 0.01, 0.0001, 0.01, 0.01]
    standard = [10, 0.001, 0.001, 10, 0.001, 0.001]
    # standard = [10, 0.01, 0.01, 10, 0.01, 0.01]
    # standard = [0.005, 0.005, 0.005, 0.005, 0.005, 0.005]

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
