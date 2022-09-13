import time

import numpy as np

from abstract.b2.b2_abs import state_space, Agent, train_model, initial_intervals, evaluate, evaluate_trace

from verify.b2.b2_env import B2_Env2

from verify.divide_tool import initiate_divide_tool_rtree, initiate_divide_tool

initial_intervals = [0.1, 0.1]
# divide_tool = initiate_divide_tool_rtree(state_space, initial_intervals, [0, 1], file_name)
divide_tool = initiate_divide_tool(state_space, initial_intervals)
agent = Agent(divide_tool)
# train_model(agent)
agent.load()
evaluate(agent)
# trace_list = evaluate_trace(agent)
# np.save('./traces/b2_trace_list', arr=trace_list)

b2 = B2_Env2(divide_tool, agent.actor)
interval_num_agg = []
r = [0.7, 0.7, 0.71, 0.71]
r = [0.7, 0.7, 0.9, 0.9]
t = 0
st = time.time()
bound_list = [r]

while True:
    t0 = time.time()
    bound_list = b2.get_next_bound_list(bound_list)
    min_x1 = 100
    max_x1 = -100
    min_x2 = 100
    max_x2 = -100

    for bound in bound_list:
        min_x1 = min(bound[0], min_x1)
        max_x1 = max(bound[2], max_x1)
        min_x2 = min(bound[1], min_x2)
        max_x2 = max(bound[3], max_x2)

    t1 = time.time()
    # print(t, '：', r[2], r[6])
    print(t, '：', len(bound_list), min_x1, max_x1, min_x2, max_x2, t1 - t0)
    interval_num_agg.append(len(bound_list))
    t += 1
    if t == 15:
        break
et = time.time()
np.save('b2_interval_number_no_agg', arr=interval_num_agg)
print(et - st)
