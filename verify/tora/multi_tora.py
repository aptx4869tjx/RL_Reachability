import multiprocessing
import time

from abstract.tora.tora_abs import *
from verify.divide_tool import initiate_divide_tool
from verify.tora.tora_env import TORAEnv1


# from verify.tora.tora_env2 import TORAEnv2


def task(region, cnt, info=False):
    tora.timer_reset()
    r = str_to_list(region)
    t = 0
    bound_list = [r]
    info_list = [[r[0], r[1], r[4], r[5]]]
    while True:
        bound_list = tora.get_next_bound_list(bound_list)

        if info:
            min_x1 = 100
            max_x1 = -100
            min_x2 = 100
            max_x2 = -100

            for bound in bound_list:
                min_x1 = min(bound[0], min_x1)
                max_x1 = max(bound[4], max_x1)
                min_x2 = min(bound[1], min_x2)
                max_x2 = max(bound[5], max_x2)
            if len(bound_list) != 0:
                info_list.append([min_x1, min_x2, max_x1, max_x2])

        t += 1
        if t == 250:
            if len(bound_list) == 0:
                print('verified', cnt)
                if info:
                    np.save('./his/tora_box_history' + str(cnt), arr=np.array(info_list))
                return True, tora.time_seg, tora.time_op, tora.time_agg
            else:
                print('exception', region, cnt, len(bound_list))
                if info:
                    np.save('./his/tora_box_history' + str(cnt), arr=np.array(info_list))
                return False, tora.time_seg, tora.time_op, tora.time_agg


if __name__ == "__main__":
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    t0 = time.time()
    sr = [[-0.77, -0.45, 0.51, -0.3], [-0.75, -0.43, 0.54, -0.28]]
    sr1 = [-0.77, -0.45, 0.51, -0.3, -0.75, -0.43, 0.54, -0.28]
    # initial_intervals = [0.2, 0.2, 0.2, 0.2]
    # initial_intervals = [0.3, 0.3, 0.3, 0.3]
    # initial_intervals = [0.4, 0.4, 0.4, 0.4]
    # initial_intervals = [0.5, 0.5, 0.5, 0.5]
    # initial_intervals = [0.1, 0.1, 0.1, 0.1]
    # initial_intervals = [0.05, 0.05, 0.05, 0.05]
    # initial_intervals = [0.15, 0.15, 0.15, 0.15]

    # initial_intervals = [0.2, 0.2, 0.1, 0.1]
    divide_tool = initiate_divide_tool(state_space, initial_intervals)
    agent = Agent(divide_tool)
    #
    # train_model(agent)
    agent.load()
    # evaluate(agent)
    tora = TORAEnv1(divide_tool, agent)
    # [0.001, 0.001, 0.015, 0.02]
    sr_dt = initiate_divide_tool(sr, [0.001, 0.001, 0.015, 0.02])
    bounds = sr_dt.intersection(sr1)
    print(len(bounds))
    pool = multiprocessing.Pool(processes=1)
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

# high = np.array([-0.75, -0.43, 0.54, -0.28])
# low = np.array([-0.77, -0.45, 0.51, -0.3])

# r = [-0.77, -0.45, 0.51, -0.3, -0.75, -0.43, 0.54, -0.28]
# r = [-0.77, -0.45, 0.51, -0.3, -0.768, -0.448, 0.512, -0.298]
