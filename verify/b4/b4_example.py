import time

import numpy as np

from abstract.b4.b4_abs import state_space, Agent, train_model, initial_intervals, evaluate, evaluate_trace

from verify.b4.b4_env import B4_Env

from verify.divide_tool import initiate_divide_tool

# initial_intervals = [0.1, 0.1, 0.1]
# initial_intervals = [0.2, 0.2, 0.2]
# divide_tool = initiate_divide_tool_rtree(state_space, initial_intervals, [0, 1], file_name)
divide_tool = initiate_divide_tool(state_space, initial_intervals)
agent = Agent(divide_tool)
# train_model(agent)
agent.load()
evaluate(agent)

# trace_list = evaluate_trace(agent)
# np.save('./traces/b4_trace_list', arr=trace_list)

b4 = B4_Env(divide_tool, agent.actor)

# high = np.array([0.27, 0.27, 0.1])
# low = np.array([0.25, 0.25, 0.08])

r = [0.25, 0.08, 0.25, 0.27, 0.1, 0.27]

# r = [0.25, 0.25, 0.08, 0.26, 0.26, 0.09]
# r = [0.26, 0.26, 0.085, 0.262, 0.262, 0.086]

t = 0
st = time.time()
bound_list = [r]
interval_num_agg = []
while True:
    t0 = time.time()
    bound_list = b4.get_next_bound_list(bound_list)
    min_x1 = 100
    max_x1 = -100
    min_x2 = 100
    max_x2 = -100
    min_x3 = 100
    max_x3 = -100

    for bound in bound_list:
        min_x1 = min(bound[0], min_x1)
        max_x1 = max(bound[3], max_x1)
        min_x2 = min(bound[1], min_x2)
        max_x2 = max(bound[4], max_x2)
        min_x3 = min(bound[2], min_x3)
        max_x3 = max(bound[5], max_x3)

    t1 = time.time()
    # print(t, '：', r[2], r[6])
    print(t, '：', len(bound_list), min_x1, max_x1, min_x2, max_x2, min_x3, max_x3, t1 - t0)
    interval_num_agg.append(len(bound_list))
    t += 1
    if t == 60:
        break
et = time.time()
np.save('b4_interval_number_agg', arr=interval_num_agg)
print(et - st)
