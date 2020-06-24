# coding: UTF-8

import random
from random import randint
import time
import collections
import itertools as it
import numpy as np
from ..help import combination, tool
from ..help import log


class AcoStaffing:

    def __init__(self, staff=None, skill=None, task=None, staff_skill=None, task_skill=None, task_precedent=None, mind_strategy=None):
        """
        Implementación de Ant Colony Optimization para asignación de personal a las actividades
        de un proyecto, sujeto:
          * Ningun trabajador debe estar sobrecargado al mismo tiempo, es decir,
            la suma de toda dedicación a sus tareas asignadas debe ser máximo el 100% de su dedicaión.
          * Todas los skill de las tareas debe ser cubiertos:
          * Seleccionar aquellas soluciones (asignaciones de personal) de mínimo costo (salario) y duración.
            (más adelante se considererar la calidad)
        El  algoritmo ACO es la generación ACO prueba uno a la vez.
        El recorrido de una hormiga debe ser tal que cada caso de prueba no cubierto deberá ser explorado
        """
        # cada valor del parámetro representa la cantidad de variables que puede aceptar o cambiar.
        self.__staff = staff
        self.__skill = skill
        self.__task = task
        self.__staff_skill = staff_skill
        self.__task_skill = task_skill
        self.__task_precedent = task_precedent
        self.__mind_strategy = mind_strategy
        self.config(ants=20, alpha=3, beta=1, rho=0.5, tau=0.4, quu=0.5, generation=1000)
        #self.weight_config(wcost=10**(-6), wdur=10**(-1), wover=10**(-1))
        self.weight_config(wcost=0.1, wdur=0.9, wover=0.9)

    def config(self, ants=None, alpha=None, beta=None, rho=None, tau=None, quu=None, generation=None):
        self.__ants = ants              # número de hormigas
        self.__alpha = alpha            # coeficiente para el control de la influencia/peso de la cantidad de feromonas
        self.__beta = beta              # coeficiente para el control de la influencia/peso de la inversa (una ruta/distancia) de la distancia
        self.__rho = rho                # tasa de volatilidad de las feromonas
        self.__tau = tau                # valor inicial de la feromona
        self.__quu = quu                # Valor que permite la explotación o exploración de nuevas rutas (regla de proporcionalidad aleatoria)
        self.__generation = generation  # máximo de generaciones (número de veces) que hará el recorrido de todas hormiga

    def weight_config(self, wcost=None, wdur=None, wover=None):
        self.__wcost = wcost            # Peso de importancia del costo
        self.__wdur = wdur              # Peso de importancia de la duración
        self.__wover = wover            # Peso de importancia del sobretiempo

    def run(self):
        # 00. Generación de la matriz con con el división de cada tareas, dada por la densidad de la dedicación
        #     den = 1 + 1/mind ó len(matrix_mind).numrow
        task_paths = self.generate_task_split()
        # print(task_paths)

        # 02. Inicializar el valor feromonas
        pheromone_values = self.generate_pheromone_values()
        # print(pheromone_values)

        # 03. Inicializar el valor feromonas
        # heuristic_values = self.generate_heuristic_values()
        # print("---", heuristic_values)

        # 03. Generar solución aleatoria
        current_solution, current_goals = self.generate_initial_solution()

        print("Solucion inicial:")
        self.result_print(current_solution, current_goals)

        last_ind = 0.0
        last_ind_new = 0.0
        ini_force_close_time = time.process_time()

        iteration = 0
        max_wait = 100
        count_wait = 0
        # self.__generation = 1
        # self.__ants = 1
        # print(pheromone_values)

        while (count_wait <= max_wait) and (iteration <= self.__generation):
            iteration += 1
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
                    # print("PPPUNO", pheromone_values[i],[np.argmax(node) for node in ant_values[i]])
                    alloc_dedicate = alloc_dedicate + [self.__mind_strategy[np.argmax(node)][1] for node in ant_values[i]]
                    # print("alloc_dedicate", iteration, _, i, alloc_dedicate, heuristic_values)
                    pheromone_values[i] = self.local_pheromones_update(pheromone_values[i], [np.argmax(node) for node in ant_values[i]], i)

                candidate_solution = np.asarray([[self.__mind_strategy[np.argmax(node)][1] for node in staff] for staff in ant_values])
                if self.assess_feasibility(candidate_solution):
                    candidate_solution, candidate_goals = self.compute_candidate_solution(candidate_solution)
                    if candidate_goals[0] > current_goals[0]:
                        print("Mejor objetivo", iteration, _, candidate_goals[0])
                        current_solution = candidate_solution
                        current_goals = candidate_goals
                        # Update feromona de forma global
                        pheromone_values = self.global_pheromones_update(pheromone_values, [[np.argmax(node) for node in staff] for staff in ant_values], candidate_goals)

        print("Solucion Final:")
        # print(pheromone_values)
        self.result_print(current_solution, current_goals)

    # FUNCIONES DE APOYO

    def exploit_ant_path(self, pheromone_values, heuristic_values):
        up = (pheromone_values ** self.__alpha) * (heuristic_values ** self.__beta)
        down = [[u if u > 0 else 1.0] for u in np.sum(up, axis=1)]
        # random_exploit = np.asarray([np.asarray(np.random.uniform(0, 1, len(self.__mind_strategy))) for _ in range(len(self.__staff))])
        # return random_exploit*(up/down)
        return up/down

    def explore_ant_path(self, heuristic_values):
        up = (heuristic_values ** self.__beta)
        down = [[u if u > 0 else 1.0] for u in np.sum(up, axis=1)]
        random_explore = np.asarray([np.asarray(np.random.uniform(0, 1, len(self.__mind_strategy))) for _ in range(len(self.__staff))])
        return random_explore*(up/down)

    # Genera la división de la tarea en el número de las estrategías de división
    def generate_task_split(self):
        return self.generate_staff_task_matrix(np.asarray([g[1] for _, g in enumerate(self.__mind_strategy)]))

    def generate_ant_values(self):
        return self.generate_staff_task_matrix(np.asarray([0.0 for _ in range(len(self.__mind_strategy))]))

    def generate_pheromone_values(self):
        return self.generate_staff_task_matrix(np.asarray([0.0 for _ in range(len(self.__mind_strategy))]))

    # def generate_heuristic_values(self):
    #     total=np.sum(self.__mind_strategy[:, 1])
    #     return self.generate_staff_task_matrix(np.asarray([1/total for _ in range(len(self.__mind_strategy))]))
    #     strategy=np.asarray([1/total for _ in range(len(self.__mind_strategy))])

    def heuristic_values(self, alloc_dedicate, task):
        # return np.asarray([[0.4 for g in self.__mind_strategy] for i in range(len(self.__staff))])
        task_skill = self.__task_skill[task]
        heuristic = np.asarray([[g[1] for g in self.__mind_strategy] for i in range(len(self.__staff))])

        for i, e in enumerate(heuristic):
            alloc = alloc_dedicate[i]
            if(alloc > 0.5):
                heuristic[i] = e + alloc_dedicate[i] - 0.5
            else:
                heuristic[i] = e + alloc_dedicate[i]

        total_tmp = np.sum(heuristic, axis=1)

        for i, e in enumerate(heuristic):
            total = total_tmp[i]
            alloc = alloc_dedicate[i]
            if(alloc > 0.5):
                heuristic[i] = (e/total)[::-1]
            else:
                heuristic[i] = e/total

        # total = np.sum(self.__mind_strategy[:, 1])
        # return self.generate_staff_task_matrix(np.asarray([1/total for _ in range(len(self.__mind_strategy))]))
        # strategy = np.asarray([1/total for _ in range(len(self.__mind_strategy))])
        return np.asarray(heuristic)

    def generate_staff_task_matrix(self, strategy):
        return np.asarray([[strategy for _ in range(len(self.__staff))] for _ in range(len(self.__task))])
        # staff_task_matrix = []
        # for i in range(len(self.__task)):
        #     task_strategy = []
        #     for j in range(len(self.__staff)):
        #         task_strategy.append(strategy)
        #     staff_task_matrix.append(np.asarray(task_strategy))

        # return np.asarray(staff_task_matrix)

    def generate_initial_solution(self):
        solution_matrix = []
        while True:
            solution_matrix = []
            for i in range(len(self.__task)):
                ded = []
                for j in range(len(self.__staff)):
                    k = randint(0, len(self.__mind_strategy) - 1)
                    ded.append(self.__mind_strategy[k][1])
                solution_matrix.append(ded)

            if self.assess_feasibility(solution_matrix):
                break

        return self.compute_candidate_solution(np.asarray(solution_matrix))

    def compute_candidate_solution(self, solution_matrix):

        # Calculando duración del proyecto
        project_cost_total, task_matrix_dur = self.project_cost_calc(solution_matrix)

        # Caculando el inicio y el final de las actividades
        project_dur_total, tpg_scheduler = self.project_duration_calc(task_matrix_dur)

        # Caculando el inicio y el final maximo de las actividades
        project_max_dur_total, tpg_max_scheduler = self.project_max_duration_calc()

        # Calculando trabajo de sobretiempo en el staff y tareas
        project_overtime_task_total, project_overeffort_staff_total = self.project_over_calc(task_matrix_dur, tpg_scheduler, solution_matrix)

        fitness_value = self.fitness_function(project_cost_total,
                                              project_dur_total,
                                              project_overtime_task_total,
                                              project_overeffort_staff_total)

        return solution_matrix, np.asarray([fitness_value,
                                            project_cost_total,
                                            project_dur_total,
                                            project_max_dur_total,
                                            project_overtime_task_total,
                                            project_overeffort_staff_total,
                                            np.asarray(task_matrix_dur),
                                            np.asarray(tpg_scheduler)])

    def fitness_function(self, cost, duration, overtime, overeffort):
        fitness_pre = self.__wcost*cost + self.__wdur*duration + self.__wover*overtime + self.__wover*overeffort
        return 0 if fitness_pre <= 0 else (fitness_pre**-1)

    def assess_feasibility(self, solution_matrix):
        # Validando que cada tarea este a cargo de mínimo un empleado
        if np.min([np.sum(v) for _, v in enumerate(solution_matrix)]) > 0:
            # for i in range(len(self.__task)):
            #     a = self.__task_skill[i, 1:]
            #     c = [0 for _ in range(len(a))]
            #     for j, val in enumerate(solution_matrix[i]):
            #         if val > 0:
            #             b = self.__staff_skill[j, 1:]
            #             c = [int(c[k] or b[k]) for k in range(len(b))]
            #             d = [int(a[k] and b[k]) for k in range(len(b))]
            #             # Validando que las personas asignadas tegna mínimo una habilidad requeridad pora tarea
            #             if np.sum(d) > 0:
            #                 return False
            #     e = [int(c[k] and a[k]) for k in range(len(a))]
            #     # Validando que las personas asignadas a una tarea cubran las habilidades requeridad (skills)
            #     if np.sum(a) != np.sum(e):
            #         return False

            # Validando que las personas asignadas a una tarea cubran las habilidades requeridad (skills)
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

    def project_cost_calc(self, solution_matrix):
        # Calculando duración del proyecto
        # cost_dur_total = 0
        project_cost_total = 0
        task_matrix_dur = []
        for i in range(len(self.__task)):
            # staff_dur = np.sum([e[2]*solution_matrix[i][j] for j, e in enumerate(self.__staff)])
            staff_dur = np.sum([solution_matrix[i][j] for j in range(len(self.__staff))])
            task_dur = (0 if staff_dur <= 0 else (self.__task[i][1] / staff_dur))
            task_matrix_dur.append(task_dur)
            # cost_dur_total += task_dur
            project_cost_total += np.sum([e[1]*solution_matrix[i][j]*task_dur for j, e in enumerate(self.__staff)])

        return project_cost_total, task_matrix_dur

    def project_duration_calc(self, task_matrix_dur):
        # Caculando el inicio y el final de las actividades
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
            # print(i, precedents, start, end, tpg_scheduler)
            project_dur_total = end
        return project_dur_total, tpg_scheduler

    def project_max_duration_calc(self):
        # Caculando el inicio y el final maximo de las actividades
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
        # Calculando trabajo de sobretiempo en el staff y tareas
        # De tareas
        project_overtime_task_total = 0
        for i, v in enumerate(task_matrix_dur):
            time_max = self.__task[i][2]
            # print(v-time_max)
            if v > time_max:
                project_overtime_task_total += (v-time_max)

        # De personas
        # Mejorar el cálculo para actividade que se sobrepones, usando una unidad de tiempo (por ejemplo días)
        tpg_scheduler_tmp = []
        project_overeffort_staff_total = 0
        for i, e in enumerate(self.__staff):
            tpg_scheduler_tmp = np.copy(tpg_scheduler)
            over_effort = 0
            for j in range(len(self.__task)):
                end = tpg_scheduler_tmp[j][1]
                if end - tpg_scheduler_tmp[j][0] > 0:
                    over_effort += solution_matrix[j][i]
                    for k in range(j+1, len(self.__task)):
                        if end - tpg_scheduler_tmp[k][0] > 0:
                            dif = min(end - tpg_scheduler_tmp[k][0], task_matrix_dur[k])
                            tpg_scheduler_tmp[k][0] = tpg_scheduler_tmp[k][0] + dif
                            over_effort += solution_matrix[k][i]
            over_effort = over_effort - self.__task[i][2]
            project_overeffort_staff_total += 0 if over_effort <= 0 else over_effort

        return project_overtime_task_total, project_overeffort_staff_total

    def local_pheromones_update(self, pheromone, path, task):
        delta = self.delta_pheromone_local(path, task)
        for idx, edge in enumerate(path):
            for var in range(len(pheromone[idx])):
                if var == edge:
                    pheromone[idx][var] = (1 - self.__rho) * pheromone[idx][var] + self.__rho*delta
        return pheromone

    def global_pheromones_update(self, pheromone, solution, candidate_goals):
        delta = self.delta_pheromone_global(candidate_goals[1], candidate_goals[2], candidate_goals[4], candidate_goals[5])
        for i in range(len(self.__task)):
            path = solution[i]
            for idx, edge in enumerate(path):
                for var in range(len(pheromone[i][idx])):
                    if var == edge:
                        pheromone[i][idx][var] = (1 - self.__rho) * pheromone[i][idx][var] + self.__rho*delta
        return pheromone

    def delta_pheromone_local(self, path, task):
        solution_matrix = [self.__mind_strategy[edge][1] for edge in path]
        staff_dur = np.sum(solution_matrix)
        task_dur = (0 if staff_dur <= 0 else (self.__task[task][1] / staff_dur))
        task_cost = np.sum([e[1]*solution_matrix[j]*task_dur for j, e in enumerate(self.__staff)])
        over_task = (task_dur-self.__task[task][2]) if task_dur > self.__task[task][2] else 0
        fitness_pre = self.__wcost*task_cost + self.__wdur*task_dur + self.__wover*over_task
        return 0 if fitness_pre <= 0 else (fitness_pre**-1)

    def delta_pheromone_global(self, project_cost, porject_dur, project_over_task, project_over_staff):
        fitness_pre = self.__wcost*project_cost + self.__wdur*porject_dur + self.__wover*project_over_task + self.__wover*project_over_staff
        return 0 if fitness_pre <= 0 else (fitness_pre**-1)

    def result_print(self, solution, goal):
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
        print(goal[4])
        print(20*"*", "SOBRETIEMPO DE PERSONAL", 20*"*")
        print(goal[5])
        print(20*"*", "DURACION DE CADA TAREA", 20*"*")
        print(goal[6])
        print(20*"*", "RUTA CRITICA DE TAREAS (CRONOGRAMA)", 20*"*")
        print(goal[7])
        print(20*"*", 60*"*", 20*"*")
        print(20*"*", 60*"*", 20*"*")
        print(20*"*", 60*"*", 20*"*")
        # FIN:FUNCIONES DE APOYO
