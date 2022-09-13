import multiprocessing
import time

from abstract.qmpc.qmpc_abs import *
from verify.divide_tool import initiate_divide_tool, max_min_clip, list_to_str

# exit('修改为dqn1')
from verify.qmpc.qmpc_env import QMPCEnv1


def task(region, cnt, info=False):
    if isinstance(region, str):
        r = str_to_list(region)
    else:
        r = region
    t = 0
    bound_list = [r]
    info_list = [[r[0], r[1], r[6], r[7]]]
    while True:
        bound_list = qmpc.get_next_bound_list(bound_list)
        if info:
            min_x1 = 100
            max_x1 = -100
            min_x2 = 100
            max_x2 = -100
            for bound in bound_list:
                min_x1 = min(bound[0], min_x1)
                max_x1 = max(bound[6], max_x1)
                min_x2 = min(bound[1], min_x2)
                max_x2 = max(bound[7], max_x2)
            if len(bound_list) != 0:
                info_list.append([min_x1, min_x2, max_x1, max_x2])

        if not check(bound_list):
            print('exception', region, cnt, len(bound_list))
            return False
        t += 1
        if t == 120:
            if check(bound_list):
                if info:
                    np.save('./his/qmpc_box_history' + str(cnt), arr=np.array(info_list))
                print('verified', cnt)
                return True
            else:
                if info:
                    np.save('./his/qmpc_box_history' + str(cnt), arr=np.array(info_list))
                print('exception', region, cnt, len(bound_list))
                return False


def check(bound_list):
    min_x1 = 100
    max_x1 = -100
    min_x2 = 100
    max_x2 = -100
    min_x3 = 100
    max_x3 = -100
    min_x4 = 100
    max_x4 = -100
    min_x5 = 100
    max_x5 = -100
    min_x6 = 100
    max_x6 = -100

    for bound in bound_list:
        min_x1 = min(bound[0], min_x1)
        max_x1 = max(bound[6], max_x1)
        min_x2 = min(bound[1], min_x2)
        max_x2 = max(bound[7], max_x2)
        min_x3 = min(bound[2], min_x3)
        max_x3 = max(bound[8], max_x3)
        min_x4 = min(bound[3], min_x4)
        max_x4 = max(bound[9], max_x4)
        min_x5 = min(bound[4], min_x5)
        max_x5 = max(bound[10], max_x5)
        min_x6 = min(bound[5], min_x6)
        max_x6 = max(bound[11], max_x6)

    if min_x1 < -0.32 or max_x1 > 0.32 or min_x2 < -0.32 or max_x2 > 0.32 or min_x3 < -0.32 or max_x3 > 0.32:
        return False
    return True


if __name__ == "__main__":
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    t0 = time.time()
    sr = [[0.025, 0, 0, 0, 0, 0], [0.05, 0.025, 0, 0, 0, 0]]
    sr1 = [0.025, 0, 0, 0, 0, 0, 0.05, 0.025, 0, 0, 0, 0]
    initial_intervals = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    initial_intervals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    divide_tool = initiate_divide_tool(state_space, initial_intervals)
    agent = Agent(divide_tool)
    #
    # train_model(agent)
    agent.load()
    # evaluate(agent)
    qmpc = QMPCEnv1(divide_tool, agent)
    # [0.001, 0.002, 0.004, 0.004] 30 cores 680s 8000
    # [0.001, 0.002, 0.01, 0.01] 30 cores 362s 1200
    # [0.001, 0.002, 0.01, 0.01] standard = [0.002, 0.001, 0.01, 0.01] 20 cores 112s
    sr_dt = initiate_divide_tool(sr, [0.005, 0.001, 0.01, 0.01, 0.01, 0.01])
    bounds = sr_dt.intersection(sr1)
    # clip_bounds = []
    # for bou in bounds:
    #     bou = max_min_clip(str_to_list(bou), sr1)
    #     clip_bounds.append(bou)

    print(len(bounds))
    results = []
    pool = multiprocessing.Pool(processes=20)
    cnt = 1
    for bound in bounds:
        results.append(pool.apply_async(task, (bound, cnt, False)))
        cnt += 1
    pool.close()
    pool.join()
    flag = True
    for res in results:
        r = res.get()
        if not r:
            flag = False
    print('verification result:', flag)
    t1 = time.time()
    print('time cost:', t1 - t0)
