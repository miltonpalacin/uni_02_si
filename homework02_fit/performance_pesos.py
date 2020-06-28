import time
import numpy as np
from alglib.meta import aco
from alglib.help import log
import sys
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

log.COUNTER_TIME = 1
log.debug_timer("Empezando seguimiento...")

META_ACO.config(ants=20, alpha=3, beta=1, rho=0.5, tau=0.4, quu=0.5, generation=250, is_normalize=True, is_stress=True)
history = []
history_iter = []
min_total = []
fit_min = sys.maxsize
#####
cost_fit = 2180.859848
duration_fit = 151.3333333
#####

start_time_head = time.process_time()

for e in pesos:
    fit_iter = sys.maxsize
    min_iter = []
    print(40*"-")
    print("Proceso de ACO con pesos  W_Peso{0}, W_Duracion {1} para normalizar el f(cost) y el f(duration):".format(e[0], e[1]))
    print(40*"-")
    for _ in range(10):

        start_time = time.process_time()

        META_ACO.weight_config(wcost=e[0], wdur=e[1], cost_fit=cost_fit, duration_fit=duration_fit)
        solution, goal = META_ACO.run()

        end_time = time.process_time()
        dif_time = end_time - start_time

        # tiempo Fitness, Costot,Duracion, WCost, WDuration
        history.append([dif_time, goal[0], goal[1], goal[2], e[0], e[1]])

        if goal[0] <= fit_iter:
            fit_iter = goal[0]
            min_iter = [dif_time, goal[0], goal[1], goal[2], e[0], e[1]]
            print("Fitness en ITERACION:", _, "Tiempo:", dif_time, "Fitness:", goal[0], "Cost:", goal[1], "Duration:", goal[2], "WCost:", e[0], "WDuration", e[1])

    print(40*"-")
    history_iter.append(min_iter)
    if fit_iter <= fit_min:
        fit_min = fit_iter
        min_total = min_iter
        print("Fitness en PESO:", "Tiempo:", min_iter[0], "Fitness:", min_iter[1], "Cost:", min_iter[2], "Duration:", min_iter[3], "WCost:", min_iter[4], "WDuration", min_iter[5])


end_time_head = time.process_time()

print(40*"-")
print(40*"-")
print("Tiempo total utilizado: ", 7*"\t", str(end_time_head - start_time_head))
print(40*"-")
print(40*"-")
print(40*"-")
print(40*"-")
print("Fitness en MAXIMO:", "Tiempo:", min_total[0], "Fitness:", min_iter[1], "Cost:", min_total[2], "Duration:", min_total[3], "WCost:", min_total[4], "WDuration", min_total[5])
print(40*"-")
print(40*"-")
print("min_total")
print(min_total)
print(40*"-")
print("history_iter_peso")
np.savetxt("history_iter_peso.csv", history_iter, delimiter=";")
print(40*"-")
print(40*"-")
print("history_peso")
np.savetxt("history_peso.csv", history, delimiter=";")
