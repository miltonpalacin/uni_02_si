import time
import numpy as np
from alglib.meta import aco
from alglib.help import log
import input01


abc = [
    [20, 5,1,0.5,0.5,200],
    [20, 5,1,0.5,0.5,1000],
    [25, 5,1,0.5,0.5,600],
    [15, 5,1,0.5,0.5,300],
    [20, 6,1,0.5,0.5,200],
    [20, 7,1,0.5,0.5,200],
    [20, 5,1,0.8,0.1,200],
    [20, 5,1,0.6,0.2,200],
    [20, 5,1,0.2,0.8,200],
    [20, 5,1,1,0.1,200],
    [20, 5,1,0,1,200],
    [20, 5,1,0,0,200]
]

# abc = [
#     [20, 3,1,0.5,0.5,200],
#     [20, 3,2,0.5,0.5,200],
#     [20, 1,3,0.5,0.5,200],
#     [20, 4,1,0.5,0.5,200],
#     [20, 5,1,0.5,0.5,200],
#     [20, 2,1,0.5,0.5,200],
#     [10, 4,1,0.5,0.5,100],
#     [20, 4,1,0.5,0.5,400],
#     [20, 4,1,0.5,0.5,600],
#     [20, 3,1,0.7,0.2,200],
#     [20, 3,1,0.8,0.5,200],
#     [20, 3,1,0.9,0.5,200],
#     [20, 3,1,1,0.5,200],
#     [20, 3,1,0.3,0.5,200],
#     [20, 3,1,0.5,0.3,200],
#     [20, 3,1,0.5,0.4,200],
#     [20, 3,1,0.5,0.7,200],
#     [20, 3,1,0.5,0.9,200],
#     [20, 3,1,0.5,1,200],
#     [20, 3,1,0.3,0.3,200],
#     [20, 3,1,0.8,0.8,200],
#     [20, 5,2,0.7,0.7,200],
#     [40, 4,1,0.7,0.9,600],
#     [20, 1,4,0.5,0.5,300],
#     [20, 3,1,0.8,0.2,300],
#     [20, 3,1,0.8,0.2,1000],
#     [30, 5,2,0.2,0.7,1000]
# ]

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

log.COUNTER_TIME = 1
log.debug_timer("Empezando seguimiento...")

META_ACO.config(ants=20, alpha=3, beta=1, rho=0.5, tau=0.4, quu=0.5, generation=200)
history = []
history_iter = []
max_total = []
fit_max = 0

for e in abc:
    fit_iter = 0
    max_iter = []
    print(40*"-")
    print("Evaluando peso ants {0}, alpha {1}, beta {2}, rho {3}, quu {4}, generation {5}:".format(e[0], e[1], e[2], e[3], e[4], e[5]))
    print(40*"-")
    for _ in range(10):

        start_time = time.process_time()

        META_ACO.config(ants=e[0], alpha=e[1], beta=e[2], rho=e[3], tau=0.4, quu=e[4], generation=e[5])
        META_ACO.weight_config(wcost=10**(-7), wdur=10**(-1), wover=10**(-1))
        solution, goal = META_ACO.run()
        end_time = time.process_time()
        dif_time = end_time - start_time

        # tiempo Fitness, Costot,Duracion, WCost, WDuration
        history.append([dif_time, goal[0], goal[1], goal[2], e[0], e[1], e[2], e[3], e[4], e[5]])

        if fit_iter <= goal[0]:
            fit_iter = goal[0]
            max_iter = [dif_time, goal[0], goal[1], goal[2], e[0], e[1], e[2], e[3], e[4], e[5]]
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
np.savetxt("history_iter.csv", history_iter, delimiter=",")
# print(np.asarray(history_iter))
print(40*"-")
print("history")
np.savetxt("history.csv", history, delimiter=",")
#print(np.asarray(history))
print("FIN FIN FIN")

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
