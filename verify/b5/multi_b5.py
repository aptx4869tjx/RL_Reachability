import multiprocessing
import time

from abstract.b5.b5_abs import *
from verify.divide_tool import initiate_divide_tool
from verify.b5.b5_env2 import B5_Env2


def task(region, cnt, info=False):
    b5_2.timer_reset()
    r = str_to_list(region)
    t = 0
    bound_list = [r]
    info_list = [[r[0], r[1], r[3], r[4]]]
    while True:
        bound_list = b5_2.get_next_bound_list(bound_list)
        if info:
            min_x1 = 100
            max_x1 = -100
            min_x2 = 100
            max_x2 = -100

            for bound in bound_list:
                min_x1 = min(bound[0], min_x1)
                max_x1 = max(bound[3], max_x1)
                min_x2 = min(bound[1], min_x2)
                max_x2 = max(bound[4], max_x2)
            if len(bound_list) != 0:
                info_list.append([min_x1, min_x2, max_x1, max_x2])
        t += 1
        if t == 50:
            if len(bound_list) == 0:
                print('verified', cnt)
                if info:
                    np.save('./his/b5_box_history' + str(cnt), arr=np.array(info_list))
                return True, b5_2.time_seg, b5_2.time_op, b5_2.time_agg
            else:
                print('exception', region, cnt, len(bound_list))
                if info:
                    np.save('./his/b5_box_history' + str(cnt), arr=np.array(info_list))
                return False, b5_2.time_seg, b5_2.time_op, b5_2.time_agg


if __name__ == "__main__":
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    t0 = time.time()
    sr = [[0.38, 0.45, 0.25], [0.4, 0.47, 0.27]]
    sr1 = [0.38, 0.45, 0.25, 0.4, 0.47, 0.27]
    # initial_intervals = [0.1, 0.1, 0.1]
    # initial_intervals = [0.2, 0.2, 0.2]
    # initial_intervals = [0.3, 0.3, 0.3]
    # initial_intervals = [0.4, 0.4, 0.4]
    # initial_intervals = [0.05, 0.05, 0.05]
    # initial_intervals = [0.02, 0.02, 0.02]
    # initial_intervals = [0.01, 0.01, 0.01]
    divide_tool = initiate_divide_tool(state_space, initial_intervals)
    agent = Agent(divide_tool)

    # train_model(agent)
    agent.load()
    # evaluate(agent)
    b5_2 = B5_Env2(divide_tool, agent.network)

    sr_dt = initiate_divide_tool(sr, [0.01, 0.01, 0.001])
    bounds = sr_dt.intersection(sr1)
    print(len(bounds))
    pool = multiprocessing.Pool(processes=20)
    cnt = 1
    results = []
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
