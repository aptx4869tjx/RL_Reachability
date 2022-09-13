import copy
import time

from scipy.optimize import minimize

from abstract.qmpc.qmpc_abs import *

import math

from verify.divide_tool import str_to_list, list_to_str, max_min_clip, combine_bound_list, contain


class QMPCEnv1:
    def __init__(self, divide_tool, agent):
        self.initial_state = [0.001, 0.001, 0, 0.001]
        self.initial_state_region = None

        # proposition_list, limited_count, limited_depth, atomic_propositions, formula,
        #                  get_abstract_state, get_abstract_state_label, get_abstract_state_hash, rtree
        self.proposition_list = []
        self.limited_count = 500000
        self.limited_depth = 500
        self.atomic_propositions = ['safe']
        self.formula = 'not(A(G(safe)))'

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.length = 0.5
        self.force_mag = 10.0
        self.tau = 0.02
        self.total_mass = (self.masspole + self.masscart)
        self.polemass_length = (self.masspole * self.length)
        self.testnum = '0'
        self.divide_tool = divide_tool
        self.network = agent.network
        self.agent = agent

    def is_done(self):
        pass

    def get_abstract_state(self, s):
        return self.divide_tool.get_abstract_state(s)

    def get_abstract_state_hash(self, abstract_state):
        return str(abstract_state)

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
        # if current[0] > -0.32 and current[6] < 0.32 and current[1] > -0.32 and current[7] < 0.32 and current[
        #     2] > -0.32 and current[8] < 0.32:
        #     return [], 0

        target_list = self.divide_tool.intersection(current)
        # print(len(target_list))
        tl = []
        for ele in target_list:
            s = str_to_list(ele)
            s = max_min_clip(current, s)
            tl.append(s)

        b_list = []
        for t in tl:
            b = self.gn(t)
            b_list.append(b)

        res = combine_bound_list(b_list, near_bound)
        return res, len(b_list)

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

        next_bounds = self.next_abstract_domain(current, act)

        return next_bounds

    def next_abstract_domain(self, current, act):
        x1l, x2l, x3l, x4l, x5l, x6l, x1u, x2u, x3u, x4u, x5u, x6u = tuple(current)

        # u1 = np.clip(act, -1, 1)[0]
        # u2 = np.clip(act, -1, 1)[1]
        # # u1 = np.clip(act, -1, 1)[0]
        # # u2 = np.clip(act, -1, 1)[1]
        # # u3 = np.clip(act, 7.81, 11.81)[2]
        # u3 = np.clip(act, -1, 1)[2]
        ##################################################################
        u1 = act[0]
        u2 = act[1]
        u3 = act[2]
        u4 = act[3]
        u5 = act[4]
        u6 = act[5]
        u7 = act[6]
        u8 = act[7]
        f20 = u2 - u7
        f19 = u2 - u6
        f18 = u2 - u5
        f17 = u2 - u4
        f16 = u2 - u3
        f15 = u1 - u8
        f14 = u1 - u7
        f13 = u1 - u6
        f12 = u1 - u5
        f11 = u1 - u4
        f10 = u1 - u3
        f9 = u1 - u2
        d28 = u2 - u8
        d34 = u3 - u4
        d35 = u3 - u5
        d36 = u3 - u6
        d37 = u3 - u7
        d38 = u3 - u8
        d45 = u4 - u5
        d46 = u4 - u6
        d47 = u4 - u7
        d48 = u4 - u8
        d56 = u5 - u6
        d57 = u5 - u7
        d58 = u5 - u8
        d67 = u6 - u7
        d68 = u6 - u8
        d78 = u7 - u8
        ##################################################################

        if f15 <= 0 and d28 <= 0 and d38 <= 0 and d48 <= 0 and d58 <= 0 and d68 <= 0 and d78 <= 0:
            u1 = 0.1
            u2 = 0.1
            u3 = 11.81
        elif f14 <= 0 and f20 <= 0 and d37 <= 0 and d47 <= 0 and d57 <= 0 and d67 <= 0 and d78 >= 0:
            u1 = 0.1
            u2 = 0.1
            u3 = 7.81
        elif f13 <= 0 and f19 <= 0 and d36 <= 0 and d46 <= 0 and d56 <= 0 and d67 >= 0 and d68 >= 0:
            u1 = 0.1
            u2 = -0.1
            u3 = 11.81
        elif f12 <= 0 and f18 <= 0 and d35 <= 0 and d45 <= 0 and d56 >= 0 and d57 >= 0 and d58 >= 0:
            u1 = 0.1
            u2 = -0.1
            u3 = 7.81
        elif f11 <= 0 and f17 <= 0 and d34 <= 0 and d45 >= 0 and d46 >= 0 and d47 >= 0 and d48 >= 0:
            u1 = -0.1
            u2 = 0.1
            u3 = 11.81
        elif f10 <= 0 and f16 <= 0 and d34 >= 0 and d35 >= 0 and d36 >= 0 and d37 >= 0 and d38 >= 0:
            u1 = -0.1
            u2 = 0.1
            u3 = 7.81
        elif f9 <= 0 and f16 >= 0 and f17 >= 0 and f18 >= 0 and f19 >= 0 and f20 >= 0 and d28 >= 0:
            u1 = -0.1
            u2 = -0.1
            u3 = 11.81
        elif f9 >= 0 and f10 >= 0 and f11 >= 0 and f12 >= 0 and f13 >= 0 and f14 >= 0 and f15 >= 0:
            u1 = -0.1
            u2 = -0.1
            u3 = 7.81
        else:
            u1 = -0.1
            u2 = -0.1
            u3 = 7.81
            print('no case!')

        u1 = round(math.tan(u1), 10)
        u2 = round(math.tan(u2), 10)
        u3 = u3 - 9.81

        t = 0.05
        # x1_new = x1 + (x4 - 0.25) * t
        # x2_new = x2 + (x5 + 0.25) * t
        # x3_new = x3 + x6 * t
        # x4_new = x4 + (9.81 * u1) * t
        # x5_new = x5 + (-9.81 * u2) * t
        # x6_new = x6 + (u3 - 9.81) * t

        nx1l = round(x1l + (x4l - 0.25) * t, 5)
        nx1u = round(x1u + (x4u - 0.25) * t, 5)
        nx2l = round(x2l + (x5l + 0.25) * t, 5)
        nx2u = round(x2u + (x5u + 0.25) * t, 5)
        nx3l = round(x3l + x6l * t, 5)
        nx3u = round(x3u + x6u * t, 5)
        nx4l = round(x4l + (9.81 * u1) * t, 5)
        nx4u = round(x4u + (9.81 * u1) * t, 5)
        nx5l = round(x5l + (-9.81 * u2) * t, 5)
        nx5u = round(x5u + (-9.81 * u2) * t, 5)
        nx6l = round(x6l + u3 * t, 5)
        nx6u = round(x6u + u3 * t, 5)

        nx1l = np.clip(nx1l, -2, 2)
        nx1u = np.clip(nx1u, -2, 2)
        nx2l = np.clip(nx2l, -2, 2)
        nx2u = np.clip(nx2u, -2, 2)
        nx3l = np.clip(nx3l, -2, 2)
        nx3u = np.clip(nx3u, -2, 2)
        # nx4l = np.clip(nx4l, 0, 0.5)
        # nx4u = np.clip(nx4u, 0, 0.5)
        # nx5l = np.clip(nx5l, -0.5, 0)
        # nx5u = np.clip(nx5u, -0.5, 0)
        # nx6l = np.clip(nx6l, -0.25, 0.25)
        # nx6u = np.clip(nx6u, -0.25, 0.25)
        nx4l = np.clip(nx4l, -2, 2)
        nx4u = np.clip(nx4u, -2, 2)
        nx5l = np.clip(nx5l, -2, 2)
        nx5u = np.clip(nx5u, -2, 2)
        nx6l = np.clip(nx6l, -2, 2)
        nx6u = np.clip(nx6u, -2, 2)

        return [nx1l, nx2l, nx3l, nx4l, nx5l, nx6l, nx1u, nx2u, nx3u, nx4u, nx5u, nx6u]

    def get_next_bound_list(self, bound_list):
        res_list = []
        cnt = 0
        for bound in bound_list:
            next_bound_list, counter = self.get_next_states(bound)
            cnt += counter
            for next_bound in next_bound_list:
                res_list.append(next_bound)
        u = combine_bound_list(res_list, near_bound)
        # print(cnt)
        return u


# 30 : 1 ( -0.3183599935680623 -0.303799997076392 ) ( -0.15880296998322008 -0.15595297217592596 ) ( 0.2549999994277947 0.2749999999999999 )
# 判断两个bound能否合并
def near_bound(current, target):
    standard = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]  # relu3_100 relu2_20 tanh 3_100 tanh 2_20

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
