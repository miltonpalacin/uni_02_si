from alglib.help import listdict
import numpy as np


def input():
    path: "D:/Maestria/CodigoMilton/uni_02_si/homework02/input01/"
    response = listdict.PropDict({})
    path = "D:/Maestria/CodigoMilton/uni_02_si/homework02/input01/"
    response.staff = np.array([tuple(map(float, line.split(' '))) for line in open(path + "01.staff.txt").read().splitlines()])
    response.skill = np.array([tuple(map(float, line.split(' '))) for line in open(path + "02.skill.txt").read().splitlines()])
    response.task = np.array([tuple(map(float, line.split(' '))) for line in open(path + "03.task.txt").read().splitlines()])
    response.staff_skill = np.array([tuple(map(float, line.split(' '))) for line in open(path + "04.staffskill.txt").read().splitlines()])
    response.task_skill = np.array([tuple(map(float, line.split(' '))) for line in open(path + "05.taskskill.txt").read().splitlines()])
    response.task_precedent = np.array([tuple(map(float, line.split(' '))) for line in open(path + "06.taskprecedent.txt").read().splitlines()])
    response.mind_strategy = np.array([tuple(map(float, line.split(' '))) for line in open(path + "07.mindstrategy.txt").read().splitlines()])
    response.staff_desc = np.array([line.split(',') for line in open(path + "08.staffdesc.txt").read().splitlines()])
    response.task_desc = np.array([line.split(',') for line in open(path + "09.taskdesc.txt").read().splitlines()])
    return response
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
