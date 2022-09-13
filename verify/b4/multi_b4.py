import multiprocessing
import time

from abstract.b4.b4_abs import *
from verify.divide_tool import initiate_divide_tool
from verify.b4.b4_env import B4_Env


def task(region, cnt, info=False):
    b4_2.timer_reset()
    r = str_to_list(region)
    t = 0
    bound_list = [r]
    info_list = [[r[0], r[1], r[3], r[4]]]
    while True:
        bound_list = b4_2.get_next_bound_list(bound_list)
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
        if t == 60:
            if len(bound_list) == 0:
                print('verified', cnt)
                if info:
                    np.save('./his/b4_box_history' + str(cnt), arr=np.array(info_list))
                return True, b4_2.time_seg, b4_2.time_op, b4_2.time_agg
            else:
                print('exception', region, cnt, len(bound_list))
                if info:
                    np.save('./his/b4_box_history' + str(cnt), arr=np.array(info_list))
                return False, b4_2.time_seg, b4_2.time_op, b4_2.time_agg


if __name__ == "__main__":
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    t0 = time.time()
    sr = [[0.25, 0.08, 0.25], [0.27, 0.1, 0.27]]
    sr1 = [0.25, 0.08, 0.25, 0.27, 0.1, 0.27]

    # initial_intervals = [0.2, 0.2, 0.2]
    # initial_intervals = [0.3, 0.3, 0.3]
    # initial_intervals = [0.4, 0.4, 0.4]
    # initial_intervals = [0.5, 0.5, 0.5]
    # initial_intervals = [0.1, 0.1, 0.1]
    # initial_intervals = [0.05, 0.05, 0.05]
    # initial_intervals = [0.02, 0.02, 0.02]
    divide_tool = initiate_divide_tool(state_space, initial_intervals)
    agent = Agent(divide_tool)
    # initial_intervals = [0.02, 0.02] [0.01,0.01] 7s 20cores 132s 1core
    # train_model(agent)
    agent.load()
    # evaluate(agent)
    b4_2 = B4_Env(divide_tool, agent.network)

    sr_dt = initiate_divide_tool(sr, [0.01, 0.01, 0.02])
    bounds = sr_dt.intersection(sr1)
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
