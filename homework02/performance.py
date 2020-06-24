import time
import numpy as np
from alglib.meta import aco
from alglib.help import log
import input01

pesos = [
    [0.0, 1.0],
    [0.1, 0.9],
    [0.2, 0.8],
    [0.3, 0.7],
    [0.4, 0.6],
    [0.5, 0.5],
    [0.6, 0.4],
    [0.7, 0.3],
    [0.8, 0.2],
    [0.9, 0.1],
    [1.0, 0.0]
]

input = input01.input()
META_ACO = aco.AcoStaffing(
    staff=input.staff,
    skill=input.skill,
    task=input.task,
    staff_skill=input.staff_skill,
    task_skill=input.task_skill,
    task_precedent=input.task_precedent,
    mind_strategy=input.mind_strategy
)

log.COUNTER_TIME = 10
log.debug_timer("Empezando seguimiento...")

META_ACO.config(ants=20, alpha=3, beta=1, rho=0.5, tau=0.4, quu=0.5, generation=200)
history = []
history_iter = []
max_total = []
fit_max = 0

for e in pesos:
    fit_iter = 0
    max_iter = []
    print(40*"-")
    print("Evaluando peso WCost {0}, WDuration {1}:".format(e[0], e[1]))
    print(40*"-")
    for _ in range(10):

        start_time = time.process_time()

        # META_ACO.weight_config(wcost=0.1, wdur=0.9, wover=0.9)
        META_ACO.weight_config(wcost=e[0], wdur=e[1], wover=e[1])
        solution, goal = META_ACO.run()

        end_time = time.process_time()
        dif_time = end_time - start_time

        # tiempo Fitness, Costot,Duracion, WCost, WDuration
        history.append([dif_time, goal[0], goal[1], goal[2], e[0], e[1]])

        if fit_iter <= goal[0]:
            fit_iter = goal[0]
            max_iter = [dif_time, goal[0], goal[1], goal[2], e[0], e[1]]
            print("Fitness en ITERACION:", _, "Tiempo:", dif_time, "Fitness:", goal[0], "Cost:", goal[1], "Durarion:", goal[2], "WCost:", e[0], "WDuration", e[1])
    print(40*"-")
    history_iter.append(max_iter)
    if fit_max <= fit_iter:
        fit_max = fit_iter
        max_total = max_iter
        print("Fitness en PESO:", "Tiempo:", max_iter[0], "Fitness:", max_iter[1], "Cost:", max_iter[2], "Durarion:", max_iter[3], "WCost:", max_iter[4], "WDuration", max_iter[5])

print(40*"-")
print(40*"-")
print("Fitness en MAXIMO:", "Tiempo:", max_total[0], "Fitness:", max_iter[1], "Cost:", max_total[2], "Durarion:", max_total[3], "WCost:", max_total[4], "WDuration", max_total[5])
print(40*"-")
print("max_total")
print(max_total)
print(40*"-")
print("history_iter")
print(np.asarray(history_iter))
print(40*"-")
print("history")
print(np.asarray(history))

# print(60*"=")
# print(60*"=")
# print(60*"=")
# pesos = [
#     [10, 1.0],
#     [0.1, 0.9],
#     [0.2, 0.8],
#     [0.3, 0.7],
#     [0.4, 0.6],
#     [0.5, 0.5],
#     [0.6, 0.4],
#     [0.7, 0.3],
#     [0.8, 0.2],
#     [0.9, 0.1],
#     [1.0, 0.0]
# ]

# input = input01.input()
# META_ACO = aco.AcoStaffing(
#     staff=input.staff,
#     skill=input.skill,
#     task=input.task,
#     staff_skill=input.staff_skill,
#     task_skill=input.task_skill,
#     task_precedent=input.task_precedent,
#     mind_strategy=input.mind_strategy
# )

# log.COUNTER_TIME = 10
# log.debug_timer("Empezando seguimiento...")

# META_ACO.config(ants=20, alpha=3, beta=1, rho=0.5, tau=0.4, quu=0.5, generation=200)
# history = []
# history_iter = []
# max_total = []
# fit_max = 0

# for e in pesos:
#     fit_iter = 0
#     max_iter = []
#     print(40*"-")
#     print("Evaluando peso WCost {0}, WDuration {1}:".format(e[0], e[1]))
#     print(40*"-")
#     for _ in range(10):

#         start_time = time.process_time()

#         # META_ACO.weight_config(wcost=0.1, wdur=0.9, wover=0.9)
#         META_ACO.weight_config(wcost=e[0], wdur=e[1], wover=e[1])
#         solution, goal = META_ACO.run()

#         end_time = time.process_time()
#         dif_time = end_time - start_time

#         # tiempo Fitness, Costot,Duracion, WCost, WDuration
#         history.append([dif_time, goal[0], goal[1], goal[2], e[0], e[1]])

#         if fit_iter <= goal[0]:
#             fit_iter = goal[0]
#             max_iter = [dif_time, goal[0], goal[1], goal[2], e[0], e[1]]
#             print("Fitness en ITERACION:", _, "Tiempo:", dif_time, "Fitness:", goal[0], "Cost:", goal[1], "Durarion:", goal[2], "WCost:", e[0], "WDuration", e[1])
#     print(40*"-")
#     history_iter.append(max_iter)
#     if fit_max <= fit_iter:
#         fit_max = fit_iter
#         max_total = max_iter
#         print("Fitness en PESO:", "Tiempo:", max_iter[0], "Fitness:", max_iter[1], "Cost:", max_iter[2], "Durarion:", max_iter[3], "WCost:", max_iter[4], "WDuration", max_iter[5])

# print(40*"-")
# print(40*"-")
# print("Fitness en MAXIMO:", "Tiempo:", max_total[0], "Fitness:", max_iter[1], "Cost:", max_total[2], "Durarion:", max_total[3], "WCost:", max_total[4], "WDuration", max_total[5])
# print(40*"-")
# print("max_total")
# print(max_total)
# print(40*"-")
# print("history_iter"")
# print(np.asarray(history_iter))
# print(40*"-")
# print("history")
# print(np.asarray(history))
