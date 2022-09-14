import multiprocessing
import time

from abstract.b1.b1_abs import *
from verify.divide_tool import initiate_divide_tool
from verify.b1.b1_env import B1_Env


def task(region, cnt, info=False):
    # b1_2.time_op = 0
    # b1_2.time_seg = 0
    # b1_2.time_agg = 0
    b1_2.timer_reset()
    r = str_to_list(region)
    t = 0
    bound_list = [r]
    info_list = [r]
    while True:
        bound_list = b1_2.get_next_bound_list(bound_list)
        if info:
            min_x1 = 100
            max_x1 = -100
            min_x2 = 100
            max_x2 = -100

            for bound in bound_list:
                min_x1 = min(bound[0], min_x1)
                max_x1 = max(bound[2], max_x1)
                min_x2 = min(bound[1], min_x2)
                max_x2 = max(bound[3], max_x2)
            if len(bound_list) != 0:
                info_list.append([min_x1, min_x2, max_x1, max_x2])
        t += 1
        if t == 70:
            if len(bound_list) == 0:
                print('verified', cnt)
                if info:
                    np.save('./his/b1_box_history' + str(cnt), arr=np.array(info_list))
                return True, b1_2.time_seg, b1_2.time_op, b1_2.time_agg
            else:
                print('exception', region, cnt, len(bound_list))
                if info:
                    np.save('./his/b1_box_history' + str(cnt), arr=np.array(info_list))
                return False, b1_2.time_seg, b1_2.time_op, b1_2.time_agg


if __name__ == "__main__":
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

    t0 = time.time()
    sr = [[0.8, 0.5], [0.9, 0.6]]
    sr1 = [0.8, 0.5, 0.9, 0.6]
    # initial_intervals = [0.02, 0.02]
    # initial_intervals = [0.03, 0.03]
    # initial_intervals = [0.04, 0.04]
    # initial_intervals = [0.05, 0.05]
    # initial_intervals = [0.01, 0.01]
    # initial_intervals = [0.005, 0.005]
    # initial_intervals = [0.06, 0.06]
    # initial_intervals = [0.002, 0.002]
    # initial_intervals = [0.001, 0.001]
    divide_tool = initiate_divide_tool(state_space, initial_intervals)
    agent = Agent(divide_tool)
    # initial_intervals = [0.02, 0.02] [0.01,0.01] 7s 20cores 132s 1core
    # train_model(agent)
    agent.load()
    # evaluate(agent)
    b1_2 = B1_Env(divide_tool, agent.network)

    sr_dt = initiate_divide_tool(sr, [0.01, 0.01])
    bounds = sr_dt.intersection(sr1)
    print(len(bounds))
    results = []
    # time
    # cost: 14.76579761505127
    # seg
    # 11.53627324104309
    # over - app
    # 106.18299126625061
    # agg
    # 142.5310754776001
    pool = multiprocessing.Pool(processes=20)
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
