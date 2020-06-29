import time
import numpy as np
from alglib.meta import aco
from alglib.help import log
import input01

# Iniciar el control del tiempo
START_TIME = time.process_time()

# 01.INPUT-MILTON
input = input01.input()

# /***** PARA INFORME****/
print(20*"*", "PROCESAMIENTO")
print(20*"*", 60*"*", 20*"*")
print(20*"*", 60*"*", 20*"*")
print()

# Instancia de metaheuristica Ant Colony Optimization (ACO)
META_ACO = aco.AcoStaffing(
    staff=input.staff,
    skill=input.skill,
    task=input.task,
    staff_skill=input.staff_skill,
    task_skill=input.task_skill,
    task_precedent=input.task_precedent,
    mind_strategy=input.mind_strategy
)

# Realizar una prueba ejecutando el método RUN
log.COUNTER_TIME = 1
log.debug_timer("Empezando seguimiento...")


# META_ACO.config(ants=20, alpha=3, beta=1, rho=0.5, tau=0.4, quu=0.5, generation=1000,is_normalize=False, is_stress=False)
# META_ACO.weight_config(wcost=0.5, wdur=0.5)
META_ACO.config(ants=20, alpha=3, beta=1, rho=0.5, tau=0.4, quu=0.5, generation=250,is_normalize=True, is_stress=False)
META_ACO.weight_config(wcost=10**-7, wdur=10**-1, cost_fit = 2227.49798, duration_fit = 217.6131395)

solution, goal = META_ACO.run()

# Resultados
print(90*"=")
print(10 * " ", 20*"*", "MEJOR ASIGNACION DE RECURSOS", 20*"*")
print(90*"=")
print("\n")
META_ACO.result_print(solution, goal, input.task_desc, input.staff_desc)
print("\n")

# Finalizar el control del tiempo
END_TIME = time.process_time()
print("Tiempo utilizado: ", 7*"\t", str(END_TIME - START_TIME))
