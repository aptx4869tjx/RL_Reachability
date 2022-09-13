import time

from abstract.acc_6.acc6_abs import *
from verify.divide_tool import initiate_divide_tool

# exit('修改为dqn1')
from verify.acc6.acc6_env import ACC6Env1

file_name = 'cart_abs3'
print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
t0 = time.time()
# initial_intervals = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
initial_intervals = [1, 0.1, 0.1, 1, 0.1, 0.1]
# initial_intervals = [0.5, 0.05, 0.05, 0.5, 0.05, 0.05]
# initial_intervals = [0.2, 0.02, 0.02, 0.2, 0.02, 0.02]
divide_tool = initiate_divide_tool(state_space, initial_intervals)
agent = Agent(divide_tool)
#
# train_model(agent)
agent.load()
# evaluate(agent)
# trace_list = evaluate_trace(agent)
# np.save('./traces/acc_trace_list', arr=trace_list)
acc6 = ACC6Env1(divide_tool, agent)

high = np.array([91, 32.05, 0.001, 11, 30.05, 0.001])
low = np.array([90, 32, 0, 10, 30, 0])

# r = [90, 32, 0, 10, 30, 0, 90.01, 32.05, 0, 10.01, 30.05, 0]
r = [90, 32, 0, 10, 30, 0, 91, 32.05, 0, 11, 30.05, 0]
t = 0
bound_list = [r]
interval_num_agg = []
st = time.time()
while True:
    t0 = time.time()
    bound_list,_ = acc6.get_next_bound_list(bound_list)
    min_x1 = 1000
    max_x1 = -1000
    min_x2 = 1000
    max_x2 = -1000
    min_x3 = 1000
    max_x3 = -1000
    min_x4 = 1000
    max_x4 = -1000
    min_x5 = 1000
    max_x5 = -1000
    min_x6 = 1000
    max_x6 = -1000

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
    t1 = time.time()
    print(t, ':', len(bound_list), '(', min_x1, max_x1, ')', '(', min_x2, max_x2, ')', '(', min_x3, max_x3, ')', '\n (',
          min_x4, max_x4, ')', '(', min_x5, max_x5, ')', '(', min_x6, max_x6, ')',
          t1 - t0)
    interval_num_agg.append(len(bound_list))
    t += 1
    if t == 55:
        break

et = time.time()
np.save('acc_interval_number_no_agg', arr=interval_num_agg)
print(et - st)
