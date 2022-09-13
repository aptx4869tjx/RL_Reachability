import multiprocessing
import time

from abstract.acc_6.acc6_abs import *
from verify.acc6.acc6_env import ACC6Env1
from verify.divide_tool import initiate_divide_tool, str_to_list


def task(region, cnt, info=False):
    acc6.timer_reset()
    if isinstance(region, str):
        r = str_to_list(region)
    else:
        r = region
    t = 0
    bound_list = [r]
    info_list = [[r[1], r[4], r[7], r[10]]]
    while True:
        bound_list, bound_interval_list = acc6.get_next_bound_list(bound_list)
        if info:
            if len(bound_interval_list) > 0:
                for step in range(len(bound_interval_list[0])):
                    min_x1 = 1000
                    max_x1 = -1000
                    min_x4 = 1000
                    max_x4 = -1000
                    for interval_list in bound_interval_list:
                        min_x1 = min(interval_list[step][1], min_x1)
                        max_x1 = max(interval_list[step][7], max_x1)
                        min_x4 = min(interval_list[step][4], min_x4)
                        max_x4 = max(interval_list[step][10], max_x4)
                    info_list.append([min_x1, min_x4, max_x1, max_x4])
        # if info:
        #     min_x1 = 1000
        #     max_x1 = -1000
        #     min_x4 = 1000
        #     max_x4 = -1000
        #     for bound in bound_list:
        #         min_x1 = min(bound[1], min_x1)
        #         max_x1 = max(bound[7], max_x1)
        #         min_x4 = min(bound[4], min_x4)
        #         max_x4 = max(bound[10], max_x4)
        #     if len(bound_list) != 0:
        #         info_list.append([min_x1, min_x4, max_x1, max_x4])

        t += 1
        if t == 55:
            if len(bound_list) == 0:
                if info:
                    np.save('./his/acc_box_history' + str(cnt), arr=np.array(info_list))
                print('verified', cnt)
                return True, acc6.time_seg, acc6.time_op, acc6.time_agg
            else:
                if info:
                    np.save('./his/acc_box_history' + str(cnt), arr=np.array(info_list))
                print('exception', region, cnt, len(bound_list))
                return False, acc6.time_seg, acc6.time_op, acc6.time_agg


if __name__ == "__main__":
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    t0 = time.time()
    sr = [[90, 32, 0, 10, 30, 0], [91, 32.05, 0, 11, 30.05, 0]]
    sr1 = [90, 32, 0, 10, 30, 0, 91, 32.05, 0, 11, 30.05, 0]

    # initial_intervals = [1, 0.1, 0.1, 1, 0.1, 0.1]
    # initial_intervals = [2, 0.2, 0.2, 2, 0.2, 0.2]
    # initial_intervals = [3, 0.3, 0.3, 3, 0.3, 0.3]
    # initial_intervals = [4, 0.4, 0.4, 4, 0.4, 0.4]
    # initial_intervals = [0.5, 0.05, 0.05, 0.5, 0.05, 0.05]
    # initial_intervals = [0.2, 0.02, 0.02, 0.2, 0.02, 0.02]
    # initial_intervals = [0.7, 0.07, 0.07, 0.7, 0.07, 0.07]
    divide_tool = initiate_divide_tool(state_space, initial_intervals)
    agent = Agent(divide_tool)
    #
    # train_model(agent)
    agent.load()
    # evaluate(agent)
    acc6 = ACC6Env1(divide_tool, agent)

    sr_dt = initiate_divide_tool(sr, [1, 0.02, 0.01, 1, 0.02, 0.01])
    bounds = sr_dt.intersection(sr1)
    # clip_bounds = []
    # for bou in bounds:
    #     bou = max_min_clip(str_to_list(bou), sr1)
    #     clip_bounds.append(bou)

    print(len(bounds))
    results = []
    pool = multiprocessing.Pool(processes=1)
    cnt = 1
    for bound in bounds:
        results.append(pool.apply_async(task, (bound, cnt, False)))
        cnt += 1
    pool.close()
    pool.join()
    time_seg = 0
    time_op = 0
    time_agg = 0
    flag = True
    for res in results:
        r = res.get()
        if not r[0]:
            flag = False
        time_seg += r[1]
        time_op += r[2]
        time_agg += r[3]
    print('verification result:', flag)
    t1 = time.time()
    print('time cost:', t1 - t0, 'seg', time_seg, 'over-app', time_op, 'agg', time_agg)
