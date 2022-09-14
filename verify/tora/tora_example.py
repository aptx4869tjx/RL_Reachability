import time

from abstract.tora.tora_abs import *
from verify.divide_tool import initiate_divide_tool

from verify.tora.tora_env import TORAEnv1


print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
t0 = time.time()
# initial_intervals = [0.2, 0.2, 0.2, 0.2]
# initial_intervals = [0.3, 0.3, 0.3, 0.3]
# initial_intervals = [0.05, 0.05, 0.05, 0.05]
# initial_intervals = [0.1, 0.1, 0.1, 0.1]
initial_intervals = [0.2, 0.2, 0.2, 0.2]
# initial_intervals = [0.05, 0.05, 0.05, 0.05]

divide_tool = initiate_divide_tool(state_space, initial_intervals)
agent = Agent(divide_tool)
#
# train_model(agent)
agent.load()
# evaluate(agent)
# trace_list = evaluate_trace(agent)
# np.save('./traces/tora_trace_list', arr=trace_list)
tora = TORAEnv1(divide_tool, agent)

# high = np.array([-0.75, -0.43, 0.54, -0.28])
# low = np.array([-0.77, -0.45, 0.51, -0.3])

r = [-0.77, -0.45, 0.51, -0.3, -0.75, -0.43, 0.54, -0.28]
# r = [-0.77, -0.438, 0.51, -0.3, -0.769, -0.437, 0.525, -0.28]
# r = [-0.769, -0.449, 0.53, -0.3, -0.768, -0.448, 0.54, -0.28]
# # r = [-0.767, -0.449, 0.53, -0.3, -0.766, -0.448, 0.54, -0.28]
r = [-0.77, -0.449, 0.52, -0.3, -0.769, -0.448, 0.53, -0.28]
# r = [-0.768, -0.447, 0.525, -0.3, -0.766, -0.445, 0.54, -0.26]
t = 0
bound_list = [r]
interval_num_agg = []
st = time.time()
while True:
    t0 = time.time()
    bound_list = tora.get_next_bound_list(bound_list)
    min_x1 = 100
    max_x1 = -100
    min_x2 = 100
    max_x2 = -100
    min_x3 = 100
    max_x3 = -100
    min_x4 = 100
    max_x4 = -100

    for bound in bound_list:
        min_x1 = min(bound[0], min_x1)
        max_x1 = max(bound[4], max_x1)
        min_x2 = min(bound[1], min_x2)
        max_x2 = max(bound[5], max_x2)
        min_x3 = min(bound[2], min_x3)
        max_x3 = max(bound[6], max_x3)
        min_x4 = min(bound[3], min_x4)
        max_x4 = max(bound[7], max_x4)

    t1 = time.time()
    print(t, ':', len(bound_list), '(', min_x1, max_x1, ')', '(', min_x2, max_x2, ')', '(', min_x3, max_x3, ')', '(',
          min_x4, max_x4, ')', t1 - t0)
    interval_num_agg.append(len(bound_list))
    t += 1
    if t == 250:
        break

et = time.time()
# np.save('tora_interval_number_agg', arr=interval_num_agg)
print(et - st)
