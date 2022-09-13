import time

from abstract.qmpc.qmpc_abs import *
from verify.divide_tool import initiate_divide_tool

# exit('修改为dqn1')
from verify.qmpc.qmpc_env import QMPCEnv1

print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
t0 = time.time()

initial_intervals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
divide_tool = initiate_divide_tool(state_space, initial_intervals)
agent = Agent(divide_tool)
#
# train_model(agent)
agent.load()
# evaluate(agent)
trace_list = evaluate_trace(agent)
np.save('./traces/qmpc_trace_list', arr=trace_list)
# trace_list = np.load('./traces/qmpc_trace_list.npy', allow_pickle=True)
qmpc = QMPCEnv1(divide_tool, agent)


# high = np.array([0.05, 0.025, 0, 0, 0, 0])
# low = np.array([0.025, 0, 0, 0, 0, 0])

def check(bound, step):
    flag = True
    for trace in trace_list:
        s = trace[step]
        for i in range(6):
            # print(s[i], bound[i], bound[i + 6])
            if round(s[i] - bound[i], 5) >= 0 and round(bound[i + 6] - s[i], 5) >= 0:
                continue
            else:
                flag = False
                break


r = [0.025, 0, 0, 0, 0, 0, 0.05, 0.025, 0, 0, 0, 0]
# r = [0.025, 0, 0, 0, 0, 0, 0.026, 0.001, 0, 0, 0, 0]
# r = [0.025, 0, 0, 0, 0, 0, 0.03, 0.001, 0, 0, 0, 0]
# r = [0.048, 0.0115, 0, 0, 0, 0, 0.0485, 0.012, 0, 0, 0, 0]
t = 0
bound_list = [r]
st = time.time()
check_bound_list = [r]
while True:
    t0 = time.time()
    if t == 13:
        print('13')
    bound_list = qmpc.get_next_bound_list(bound_list)
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
    t1 = time.time()

    print(t, ':', len(bound_list), '(', min_x1, max_x1, ')', '(', min_x2, max_x2, ')', '(', min_x3, max_x3, ')', '\n (',
          min_x4, max_x4, ')', '(', min_x5, max_x5, ')', '(', min_x6, max_x6, ')',
          t1 - t0)
    t += 1

    check_bound = [min_x1, min_x2, min_x3, min_x4, min_x5, min_x6, max_x1, max_x2, max_x3, max_x4, max_x5, max_x6]
    check(check_bound, t)
    check_bound_list.append(check_bound)
    if t == 120:
        break

et = time.time()
print(et - st)

# [-0.18178, 0.10346, -0.01, 0.04921, -0.24605, -0.1, -0.14694, 0.12354, 0.0, 0.04921, -0.14763, 0.1]
# [-0.16827  0.1044   0.       0.04921 -0.14763 -0.1    ]
# [-0.17831  0.10952 -0.005   -0.      -0.19684 -0.     ]
