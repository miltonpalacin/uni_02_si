# coding: UTF-8

import random
from random import randint
import time
import collections
import itertools as it
import numpy as np
from ..help import combination, tool
from ..help import log
from tabulate import tabulate


class AcoStaffing:

    def __init__(self, staff=None, skill=None, task=None, staff_skill=None, task_skill=None, task_precedent=None, mind_strategy=None):
        """
        Implementacion de Ant Colony Optimization para asignación de personal a las actividades
        de un proyecto, sujeto:
          * Ningun trabajador debe estar sobrecargado al mismo tiempo, es decir,
            la suma de toda dedicación a sus tareas asignadas debe ser máximo el 100% de su dedicacion.
          * Todas los skill de las tareas debe ser cubiertos:
          * Seleccionar aquellas soluciones (asignaciones de personal) de mínimo costo (salario) y duracion.
            (más adelante se considererar la calidad)
        """
        # cada valor del parametro representa la cantidad de variables que puede aceptar o cambiar.
        self.__staff = staff
        self.__skill = skill
        self.__task = task
        self.__staff_skill = staff_skill
        self.__task_skill = task_skill
        self.__task_precedent = task_precedent
        self.__mind_strategy = mind_strategy
        # configuración incial que cambiara en las pruebas de rendimiento
        self.config(ants=20, alpha=3, beta=1, rho=0.5, tau=0.4, quu=0.5, generation=1000)
        self.weight_config(wcost=0.1, wdur=0.9)

    def config(self, ants=None, alpha=None, beta=None, rho=None, tau=None, quu=None, generation=None, is_normalize=None, is_stress=None):
        self.__ants = ants                  # numero de hormigas
        self.__alpha = alpha                # coeficiente para el control de la influencia/peso de la cantidad de feromonas
        self.__beta = beta                  # coeficiente para el control de la influencia/peso de la inversa (una ruta/distancia) de la distancia
        self.__rho = rho                    # tasa de volatilidad de las feromonas
        self.__tau = tau                    # valor inicial de la feromona
        self.__quu = quu                    # valor que permite la explotación o exploración de nuevas rutas (regla de proporcionalidad aleatoria)
        self.__generation = generation      # numero de generaciones (numero de veces) que hará el recorrido de todas las hormigas
        self.__is_normalize = is_normalize  # aplicar la normalización del costo y duración
        self.__is_stress = is_stress        # permite habiltar/deshabilitar para las pruebas de esfuerzo

    def weight_config(self, wcost=None, wdur=None, cost_fit=None, duration_fit=None):
        self.__wcost = wcost                # Peso de importancia del costo
        self.__wdur = wdur                  # Peso de importancia de la duracion
        self.__cost_fit = cost_fit          # Ajuste para el valor del costo
        self.__duration_fit = duration_fit  # Ajuste para el valor del duracion

    def run(self):

        # 01. Inicializar el valor feromonas
        pheromone_values = self.generate_pheromone_values()
        # print(pheromone_values)

        # 02. Generar solución aleatoria
        current_solution, current_goals = self.generate_initial_solution()
        if not self.__is_stress:
            self.result_print_own(current_solution, current_goals)

        # contador de generaciones
        iteration = 0

        # numero maximo de veces que se estanca la mejor solucion
        max_wait = 200
        count_wait = 0  # contador del numero maximo que se estanca la solucion

        # 03. Ejecucion principal del algoritmo
        while (count_wait <= max_wait) and (iteration <= self.__generation):
            iteration += 1
            max_wait += 1
            for _ in range(self.__ants):
                ant_values = self.generate_ant_values()
                alloc_dedicate = np.asarray([0.0 for _ in range(len(self.__staff))])
                for i in range(len(self.__task)):
                    quu = random.uniform(0, 1)
                    heuristic_values = self.heuristic_values(alloc_dedicate, i)
                    if quu > self.__quu:
                        # exploramos un nuevo ruta
                        ant_values[i] = self.explore_ant_path(heuristic_values)
                    elif quu <= self.__quu:
                        # explotamos un nuevo ruta
                        ant_values[i] = self.exploit_ant_path(pheromone_values[i], heuristic_values)

                    # Actualización local feromona
                    alloc_dedicate = alloc_dedicate + [(self.__mind_strategy[np.argmax(node)][1] * self.__staff[n][2]) for n, node in enumerate(ant_values[i])]
                    pheromone_values[i] = self.local_pheromones_update(pheromone_values[i], [np.argmax(node) for node in ant_values[i]], i)

                candidate_solution = np.asarray([[(self.__mind_strategy[np.argmax(node)][1] * self.__staff[n][2]) for n, node in enumerate(staff)] for staff in ant_values])
                if self.assess_feasibility(candidate_solution):
                    candidate_solution, candidate_goals = self.compute_candidate_solution(candidate_solution)
                    # BUSCANDO EL MÍNIMO:
                    if candidate_goals[0] < current_goals[0]:
                        max_wait = 0
                        if not self.__is_stress:
                            log.debug_timer("Mejor candidato:", "Fitness:", candidate_goals[0], "Cost:", candidate_goals[1], "Duration:", candidate_goals[2])
                        current_solution = candidate_solution
                        current_goals = candidate_goals
                        # Update feromona de forma global
                        pheromone_values = self.global_pheromones_update(pheromone_values, [[np.argmax(node) for node in staff] for staff in ant_values], candidate_goals)

        return current_solution, current_goals

    # ################################################################## #
    # ################## INI:FUNCIONES PRINCIPALES ##################### #
    # ################################################################## #

    # ACTUALIZACION LOCAL DE LA FEROMONA
    def local_pheromones_update(self, pheromone, path, task):
        delta = self.delta_pheromone_local(path, task)
        for idx, edge in enumerate(path):
            for var in range(len(pheromone[idx])):
                if var == edge:
                    pheromone[idx][var] = (1 - self.__rho) * pheromone[idx][var] + self.__rho*delta
        return pheromone

    # ACTUALIZACION LOCAL DE LA FEROMONA
    def global_pheromones_update(self, pheromone, solution, candidate_goals):
        delta = self.delta_pheromone_global(candidate_goals[1], candidate_goals[2], candidate_goals[4], candidate_goals[5], candidate_goals[8])
        for i in range(len(self.__task)):
            path = solution[i]
            for idx, edge in enumerate(path):
                for var in range(len(pheromone[i][idx])):
                    if var == edge:
                        pheromone[i][idx][var] = (1 - self.__rho) * pheromone[i][idx][var] + self.__rho*delta
        return pheromone

    # delta local de la feromona: compensa la calidad de le feromona de la solución actual (path)
    def delta_pheromone_local(self, path, task):
        solution_matrix = [self.__mind_strategy[edge][1] * self.__staff[n][2] for n, edge in enumerate(path)]

        staff_dur = np.sum(solution_matrix)
        task_dur = (0 if staff_dur <= 0 else (self.__task[task][1] / staff_dur))
        task_cost = np.sum([e[1]*solution_matrix[j]*task_dur for j, e in enumerate(self.__staff)])
        over_task = (task_dur-self.__task[task][2]) if task_dur > self.__task[task][2] else 0

        if not self.__is_normalize:
            fitness_pre = self.__wcost*task_cost + self.__wdur*(task_dur + over_task)
        else:
            fitness_pre = self.__wcost*task_cost/self.__cost_fit + self.__wdur*(task_dur + over_task)/self.__duration_fit
        return 0 if fitness_pre <= 0 else (fitness_pre**-1)

    # delta global de la feromona: compensa la calidad de le feromona de la solución de proyecto
    def delta_pheromone_global(self, project_cost, project_dur, project_over_task, project_over_staff, project_cost_ajust):
        if not self.__is_normalize:
            fitness_pre = self.__wcost*project_cost + self.__wdur*(project_dur + project_over_task + project_over_staff)
        else:
            fitness_pre = self.__wcost*project_cost/self.__cost_fit + self.__wdur*(project_dur + project_over_task + project_over_staff)/self.__duration_fit
        return 0 if fitness_pre <= 0 else (fitness_pre**-1)

    # INFORMACION DE LA HEURISTICA
    def heuristic_values(self, alloc_dedicate, task):
        heuristic = np.asarray([[g[1] * self.__staff[i][2] for g in self.__mind_strategy] for i in range(len(self.__staff))])

        # calcular el total
        for i, e in enumerate(heuristic):
            alloc = alloc_dedicate[i]
            if(alloc > 0.5):
                heuristic[i] = e + alloc_dedicate[i] - 0.5  # 50% de la dedicación
            else:
                heuristic[i] = e + alloc_dedicate[i]

        total_tmp = np.sum(heuristic, axis=1)

        task_skill = self.__task_skill[task, 1:]
        for i, e in enumerate(heuristic):
            total = total_tmp[i]
            alloc = alloc_dedicate[i]
            # calcular la heuristica de la dedicación asignada
            if(alloc > 0.5):
                heuristic[i] = (e/total)[::-1]  # invertir para permitir mas dedicacion a los otros empleados
            else:
                heuristic[i] = e/total

            # validar que el skill de staff contenga minimo uno del skill task
            staff_skill = self.__staff_skill[i, 1:]
            if np.sum([int(staff_skill[k] and task_skill[k]) for k in range(len(task_skill))]) <= 0:
                # penaliza y asignar el valor inical de la estrategia, que es 0%
                heuristic[i][0] = np.max(heuristic[i])
                for j in range(1, len(heuristic[i])):
                    heuristic[i][j] = 0

        return np.asarray(heuristic)

    # ################ #
    # ## IMPORTANTE ## #
    # ################ #
    # FUNCION DE APTITUP - FITNESS
    def fitness_function(self, cost, duration):
        # Maximizar
        if not self.__is_normalize:
            fitness_pre = self.__wcost*cost + self.__wdur*duration
        else:
            fitness_pre = self.__wcost*cost/self.__cost_fit + self.__wdur*duration/self.__duration_fit
        return 0 if fitness_pre <= 0 else fitness_pre

    # ################ #
    # ## IMPORTANTE ## #
    # ################ #

    # FUNCION DE EVALUACION DE RESTRICCIONES
    def assess_feasibility(self, solution_matrix):
        # validar que cada tarea este a cargo de minimo un empleado
        if np.min([np.sum(v) for _, v in enumerate(solution_matrix)]) > 0:
            # validar que las personas asignadas a una tarea cubran las habilidades requeridad (skills)
            for i in range(len(self.__task)):
                a = self.__task_skill[i, 1:]
                c = [0 for _ in range(len(a))]
                for j, val in enumerate(solution_matrix[i]):
                    if val > 0:
                        b = self.__staff_skill[j, 1:]
                        c = [int(c[k] or b[k]) for k in range(len(b))]
                d = [int(c[k] and a[k]) for k in range(len(a))]
                if np.sum(a) != np.sum(d):
                    return False

            return True
        return False

    # ################################################################## #
    # ################## FIN:FUNCIONES PRINCIPALES ##################### #
    # ################################################################## #

    # ################################################################## #
    # ################## FIN: FUNCIONES SECUNDARIAS #################### #
    # ################################################################## #

    def compute_candidate_solution(self, solution_matrix):
        
        # clcular la duracion del proyecto
        project_cost_total, task_matrix_dur = self.project_cost_calc(solution_matrix)

        # caclular el inicio y el final de las actividades
        project_dur_total, tpg_scheduler = self.project_duration_calc(task_matrix_dur)

        # calcular el inicio y el final maximo de las actividades
        project_max_dur_total, tpg_max_scheduler = self.project_max_duration_calc()

        # calcular trabajo de sobretiempo en el personal y tareas
        project_overtime_task_total, project_overeffort_staff_total, project_cost_ajust = self.project_over_calc(task_matrix_dur, tpg_scheduler, solution_matrix)

        fitness_value = self.fitness_function(project_cost_total, project_dur_total)

        # ajuste al costo del proyecto por el exceso de horas del personal (pago doble, la primera se dio en calculo inicial)
        # si el empleado esta en n tareas al mismo tiempo y sobre pasa su capacida se le paga n veces (este parte se tiene que MEJORAR)
        project_cost_total += project_cost_ajust

        return solution_matrix, np.asarray([fitness_value,
                                            project_cost_total,
                                            project_dur_total,
                                            project_max_dur_total,
                                            project_overtime_task_total,
                                            project_overeffort_staff_total,
                                            np.asarray(task_matrix_dur),
                                            np.asarray(tpg_scheduler),
                                            project_cost_ajust])

    def exploit_ant_path(self, pheromone_values, heuristic_values):
        up = (pheromone_values ** self.__alpha) * (heuristic_values ** self.__beta)
        down = [[u if u > 0 else 1.0] for u in np.sum(up, axis=1)]
        return up/down

    def explore_ant_path(self, heuristic_values):
        up = (heuristic_values ** self.__beta)
        down = [[u if u > 0 else 1.0] for u in np.sum(up, axis=1)]
        random_explore = np.asarray([np.asarray(np.random.uniform(0, 1, len(self.__mind_strategy))) for _ in range(len(self.__staff))])
        return random_explore*(up/down)

    def generate_ant_values(self):
        return self.generate_staff_task_matrix(np.asarray([0.0 for _ in range(len(self.__mind_strategy))]))

    def generate_pheromone_values(self):
        return self.generate_staff_task_matrix(np.asarray([0.0 for _ in range(len(self.__mind_strategy))]))

    def generate_staff_task_matrix(self, strategy):
        return np.asarray([[strategy for _ in range(len(self.__staff))] for _ in range(len(self.__task))])

    def generate_initial_solution(self):
        solution_matrix = []
        while True:
            solution_matrix = []
            for i in range(len(self.__task)):
                task_skill = self.__task_skill[i, 1:]
                ded = []
                for j in range(len(self.__staff)):

                    staff_skill = self.__staff_skill[j, 1:]
                    k = 0

                    # validar que el skill de empleado contenga minimo uno del skill task
                    if np.sum([int(staff_skill[k] and task_skill[k]) for k in range(len(task_skill))]) > 0:
                        k = randint(0, len(self.__mind_strategy) - 1)
                    ded.append(self.__mind_strategy[k][1] * self.__staff[j][2])
                solution_matrix.append(ded)

            if self.assess_feasibility(solution_matrix):
                break

        return self.compute_candidate_solution(np.asarray(solution_matrix))

    def project_cost_calc(self, solution_matrix):
        # calcular duración del proyecto
        project_cost_total = 0
        task_matrix_dur = []
        for i in range(len(self.__task)):
            #staff_dur = np.sum([e[2]*solution_matrix[i][j] for j, e in enumerate(self.__staff)])
            staff_dur = np.sum([solution_matrix[i][j] for j in range(len(self.__staff))])
            task_dur = (0 if staff_dur <= 0 else (self.__task[i][1] / staff_dur))
            task_matrix_dur.append(task_dur)
            project_cost_total += np.sum([e[1]*solution_matrix[i][j]*task_dur for j, e in enumerate(self.__staff)])

        return project_cost_total, task_matrix_dur

    def project_duration_calc(self, task_matrix_dur):
        # calcular el inicio y el final de las actividades
        tpg_scheduler = []
        project_dur_total = 0
        for i in range(len(self.__task)):
            precedents = self.__task_precedent[i, 1:]

            start = 0
            for j in range(len(self.__task)):
                if precedents[j] == 1:
                    start = max(start, tpg_scheduler[j][1])
            end = start + task_matrix_dur[i]
            tpg_scheduler.append([start, end])
            project_dur_total = end
        return project_dur_total, tpg_scheduler

    def project_max_duration_calc(self):
        # calcular el inicio y el final maximo de las actividades
        tpg_max_scheduler = []
        project_max_dur_total = 0
        for i in range(len(self.__task)):
            precedents = self.__task_precedent[i, 1:]
            start = 0
            for j in range(len(self.__task)):
                if precedents[j] == 1:
                    start = max(start, tpg_max_scheduler[j][1])
            end = start + self.__task[i][2]
            tpg_max_scheduler.append([start, end])
            project_max_dur_total = end

        return project_max_dur_total, tpg_max_scheduler

    def project_over_calc(self, task_matrix_dur, tpg_scheduler, solution_matrix):
        # calcular trabajo de sobretiempo en el staff y tareas
        # de tareas
        project_overtime_task_total = 0
        for i, v in enumerate(task_matrix_dur):
            time_max = self.__task[i][2]
            if v > time_max:
                project_overtime_task_total += (v-time_max)

        # de personas
        # MEJORAR el calculo para tareas que se sobrepones, usando una unidad de tiempo (por ejemplo días)
        # MEJORAR esta parte para que no se cuente doble, además se sugiere ponerlo dentro del fittness
        # se resuelve con una funcion recursiva, iterando la activades cruzadas hasta el final
        tpg_scheduler_tmp = []
        project_overeffort_staff_total = 0
        project_cost_ajust = 0
        for i, e in enumerate(self.__staff):
            tpg_scheduler_tmp = np.copy(tpg_scheduler)
            over_effort = 0
            cost_effort = 0
            for j in range(len(self.__task)):
                end = tpg_scheduler_tmp[j][1]
                if end - tpg_scheduler_tmp[j][0] > 0:
                    over_effort = solution_matrix[j][i]
                    for k in range(j+1, len(self.__task)):
                        if end - tpg_scheduler_tmp[k][0] > 0 and tpg_scheduler_tmp[k][1] > tpg_scheduler_tmp[k][0]:
                            dif = min(end - tpg_scheduler_tmp[k][0], task_matrix_dur[k])
                            tpg_scheduler_tmp[k][0] = tpg_scheduler_tmp[k][0] + dif
                            over_effort += solution_matrix[k][i]
                            dif_over = over_effort - e[2]  # calculando el esfuerzo en exceso
                            if dif_over > 0:
                                cost_effort += dif_over*e[1]  # calculando el costo en exceso
                                over_effort -= dif_over
            if cost_effort > 0:
                project_overeffort_staff_total += (cost_effort/e[1])
                project_cost_ajust += cost_effort

        return project_overtime_task_total, project_overeffort_staff_total, project_cost_ajust

    # ################################################################## #
    # ################## FIN: FUNCIONES SECUNDARIAS #################### #
    # ################################################################## #

    # ################################################################## #
    # ################### INI: FUNCIONES DE APOYO ###################### #
    # ################################################################## #

    # Funcion para imprimir de manera interna
    def result_print_own(self, solution, goal):
        print(20*"*", 60*"*", 20*"*")
        print(20*"*", 60*"*", 20*"*")
        print(20*"*", 60*"*", 20*"*")
        print(20*"*", "MATRIZ DE ASIGNACION", 20*"*")
        print(solution)
        print(20*"*", "VALOR FITNESS", 20*"*")
        print(goal[0])
        print(20*"*", "VALOR COST", 20*"*")
        print(goal[1])
        print(20*"*", "VALOR DURACION DEL PROYECTO", 20*"*")
        print(goal[2])
        print(20*"*", "VALOR DURACION MAXIMA", 20*"*")
        print(goal[3])
        print(20*"*", "SOBRETIEMPO DE TAREAS", 20*"*")
        print(goal[4], )
        print(20*"*", "SOBRETIEMPO DE PERSONAL", 20*"*")
        print("Esfuerzo extra:", goal[5], "Costo extra:", goal[8])
        print(20*"*", "DURACION DE CADA TAREA", 20*"*")
        print(goal[6])
        print(20*"*", "RUTA CRITICA DE TAREAS (CRONOGRAMA)", 20*"*")
        print(goal[7])
        print(20*"*", 60*"*", 20*"*")
        print(20*"*", 60*"*", 20*"*")
        print(20*"*", 60*"*", 20*"*")

    # Funcion para imprimir de manera externa
    def result_print(self, solution, goal, task_desc, staff_desc):
        print(20*"*", 60*"*", 20*"*")
        print(20*"*", 60*"*", 20*"*")
        print(20*"*", 60*"*", 20*"*")
        print(20*"*", "MATRIZ DE ASIGNACION", 20*"*")
        sol = [list(map(str, x)) for x in np.array(solution.transpose())]
        sol = [[10*" "+i for i in e] for e in sol]
        sol = np.insert(sol, 0,  task_desc[:, 1:].transpose()[0], axis=0)
        sol = np.insert(sol, 0,  np.insert(staff_desc[:, 1:], 0, " ", axis=0).transpose()[0], axis=1)

        table = tabulate(sol, headers="firstrow", tablefmt="grid")
        tabulate.PRESERVE_WHITESPACE = True
        print(table)
        print(20*"*", "VALOR FITNESS", 20*"*")
        print(goal[0])
        print(20*"*", "VALOR COST", 20*"*")
        print(goal[1])
        print(20*"*", "VALOR DURACION DEL PROYECTO", 20*"*")
        print(goal[2])
        print(20*"*", "VALOR DURACION MAXIMA", 20*"*")
        print(goal[3])
        print(20*"*", "SOBRETIEMPO DE TAREAS", 20*"*")
        print(goal[4])
        print(20*"*", "SOBRETIEMPO DE PERSONAL", 20*"*")
        print("Esfuerzo extra:", goal[5], "Costo extra:", goal[8])
        print(20*"*", "DURACION DE CADA TAREA", 20*"*")
        print(goal[6])
        print(20*"*", "RUTA CRITICA DE TAREAS (CRONOGRAMA)", 20*"*")
        print(goal[7])
        print(20*"*", 60*"*", 20*"*")
        print(20*"*", 60*"*", 20*"*")
        print(20*"*", 60*"*", 20*"*")

    # ################################################################## #
    # ################### FIN: FUNCIONES DE APOYO ###################### #
    # ################################################################## #
