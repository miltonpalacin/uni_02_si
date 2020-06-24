import time
import numpy as np
from alglib.meta import aco
from alglib.help import log

# Iniciar el control del tiempo
START_TIME = time.process_time()

# 01.INPUT-MILTON
path = "D:/Maestria/CodigoMilton/uni_02_si/homework02/input01/"
staff = np.array([tuple(map(float, line.split(' '))) for line in open(path + "01.staff.txt").read().splitlines()])
skill = np.array([tuple(map(float, line.split(' '))) for line in open(path + "02.skill.txt").read().splitlines()])
task = np.array([tuple(map(float, line.split(' '))) for line in open(path + "03.task.txt").read().splitlines()])
staff_skill = np.array([tuple(map(float, line.split(' '))) for line in open(path + "04.staffskill.txt").read().splitlines()])
task_skill = np.array([tuple(map(float, line.split(' '))) for line in open(path + "05.taskskill.txt").read().splitlines()])
task_precedent = np.array([tuple(map(float, line.split(' '))) for line in open(path + "06.taskprecedent.txt").read().splitlines()])
mind_strategy = np.array([tuple(map(float, line.split(' '))) for line in open(path + "07.mindstrategy.txt").read().splitlines()])
staff_desc = np.array([line.split(',') for line in open(path + "08.staffdesc.txt").read().splitlines()])
task_desc = np.array([line.split(',') for line in open(path + "09.taskdesc.txt").read().splitlines()])
# print()
# print(20*"*", "PARAMETROS")
# print(20*"*", 60*"*", 20*"*")
# print(20*"*", 60*"*", 20*"*")
# print()
# print(20*"*", "Matriz de empleados", 20*"*")
# print(staff)
# print(20*"*", "Matriz de habilidades", 20*"*")
# print(skill)
# print(20*"*", "Matriz de tareas del proyecto", 20*"*")
# print(task)
# print(20*"*", "Matriz de  empleados y sus habilidades", 20*"*")
# print(staff_skill)
# print(20*"*", "Matriz de tareas y las habilidades requeridas", 20*"*")
# print(task_skill)
# print(20*"*", "Matriz de precedendica de tareas (TPG => Task Precedent Graph)", 20*"*")
# print(task_precedent)
# print(20*"*", "Matriz de estrategias de dedicacion", 20*"*")
# print(mind_strategy)
# print()
# print(20*"*", 60*"*", 20*"*")
# print(20*"*", 60*"*", 20*"*")
# print()

# /***** PARA INFORME****/
print(20*"*", "PROCESAMIENTO")
print(20*"*", 60*"*", 20*"*")
print(20*"*", 60*"*", 20*"*")
print()

# Instancia de metaheuristica Ant Colony Optimization (ACO)
META_ACO = aco.AcoStaffing(
    staff=staff,
    skill=skill,
    task=task,
    staff_skill=staff_skill,
    task_skill=task_skill,
    task_precedent=task_precedent,
    mind_strategy=mind_strategy
)
# Realizar una prueba ejecutando el método RUN
log.COUNTER_TIME = 10
log.debug_timer("Empezando seguimiento...")
META_ACO.config(ants=20, alpha=3, beta=1, rho=0.5, tau=0.4, quu=0.5, generation=2)
META_ACO.weight_config(wcost=0.1, wdur=0.9, wover=0.9)
solution, goal = META_ACO.run()

# Resultados
print(90*"=")
print(10 * " ", 20*"*", "MEJOR ASIGNACION DE RECURSOS", 20*"*")
print(90*"=")
print("\n")
META_ACO.result_print(solution, goal, task_desc, staff_desc)
print("\n")

# Finalizar el control del tiempo
END_TIME = time.process_time()
print("Tiempo utilizado: ", 7*"\t", str(END_TIME - START_TIME))
