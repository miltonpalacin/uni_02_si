import time
import numpy as np
from alglib.meta import aco
from alglib.help import log
import input01

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

META_ACO.config(ants=20, alpha=3, beta=1, rho=0.5, tau=0.4, quu=0.5, generation=1000, is_normalize=False, is_stress=True)
META_ACO.weight_config(wcost=0.5, wdur=0.5)

history = []

print(40*"-")
print(40*"-")
print("Proceso de ACO con pesos  W_Peso{0}, W_Duracion {1} para normalizar el f(cost) y el f(duration):".format(0.5, 0.5))
print(40*"-")
print(40*"-")
start_time_head = time.process_time()

for _ in range(100):
    start_time = time.process_time()

    solution, goal = META_ACO.run()

    end_time = time.process_time()
    dif_time = end_time - start_time

    # tiempo Fitness, Costot,Duracion, WCost, WDuration
    history.append([dif_time, goal[0], goal[1], goal[2], 0.5, 0.5])
    log.debug_timer("Proceso de :", _, "Fitness:", goal[0], "Cost:", goal[1], "Duration:", goal[2])

end_time_head = time.process_time()

print(40*"-")
print(40*"-")
print("Tiempo total utilizado: ", 7*"\t", str(end_time_head - start_time_head))
print(40*"-")
print(40*"-")
print("Finalizacion del proceso")
print(40*"-")
print(40*"-")
print("history_normalize")
np.savetxt("history_normalize.csv", history, delimiter=";")
print(40*"-")
print(40*"-")
