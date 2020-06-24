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
        self.config(ants=20, alpha=0.5, beta=3, rho=0.5, tau=0.4, quu=0.5, generation=1000)
        self.weight_config(wcost=10**(-6), wdur=10**(-1), wover=10**(-1))

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
        heuristic_values = self.generate_heuristic_values()
        # print("---", heuristic_values)

        # 04. Generar solución aleatoria
        current_solution, goals = self.generate_initial_solution()

        print("Solucion inicial:")
        print(current_solution)
        print(goals)

        last_ind = 0.0
        last_ind_new = 0.0
        ini_force_close_time = time.process_time()

        iteration = 0
        max_wait = 100
        count_wait = 0
        self.__generation = 1
        self.__ants = 1

        while (count_wait <= max_wait) and (iteration <= self.__generation):
            iteration += 1
            for _ in range(self.__ants):
                ant_values = self.generate_ant_values()
                # candidate_solution = np.asarray([[0.0 for _ in range(len(self.__staff))] for _ in range(len(self.__task))])
                for i in range(len(self.__task)):
                    quu = random.uniform(0, 1)
                    if quu > self.__quu:
                        # exploramos un nuevo ruta
                        ant_values[i] = self.explore_ant_path(heuristic_values[i])
                    elif quu <= self.__quu:
                        # explotamos un nuevo ruta
                        ant_values[i] = self.exploit_ant_path(pheromone_values[i], heuristic_values[i])

                candidate_solution = np.asarray([[self.__mind_strategy[np.argmax(node)][1] for node in staff] for staff in ant_values])
                if self.assess_feasibility(candidate_solution):
                    continue
                print(30*"*", "candidate_solution")
                print(candidate_solution)
                print(30*"*", "compute_candidate_solution")
                print(self.compute_candidate_solution(candidate_solution))
        """
        # 1. Generar todas las posibles interacciones basado en el total de valores que pueden tomar cada uno de los parámetros
        #    V1^P1, V2^P2, V3^P3,..., Vn^Pn
        uncovering_list = AcoTWay.initial_uncovering_list(self.__parameters, self.__t_way)

        # 2. Iniciar la matriz de covertura
        covering_list = []

        # 3. Iniciar los principales argumentos del algoritmo
        #    el número de iteración y el número de hormigas se han asignado en el constructor



        # 4. Recorrer todos los casos de prueba generados  posibles
        while len(uncovering_list) > 0:
            # rutas candidatas
            candidates_path = []

            # 5. Iniciar matriz de feromonas con la constante ingresada _tau
            pheromone = AcoTWay.start_matrix_values(self.__parameters, self.__tau)

            # 6. Cálculo inicial de la heuristica para cada recorrido de la hormiga
            heuristic = AcoTWay.compute_heuristic_values(self.__parameters, covering_list)

            # Para mostrar un log en el caso quede pegado en un optimo local
            ########### INI LOG ##################
            end_force_close_time = time.process_time()
            if end_force_close_time - ini_force_close_time > 60:
                print("Caso donde falto completar la combinacion de alguna de sus variables:",uncovering_list)
                break
            log.debug_timer("Combinaciones NO cubiertas:", len(uncovering_list), ", Casos de prueba cubiertos:", len(covering_list), "indicador", last_ind)
            last_ind_new = len(covering_list)/len(uncovering_list)
            if(last_ind_new != last_ind):
                ini_force_close_time = time.process_time()
                last_ind = last_ind_new
            ########### INI FIN ##################

            # 7. Repetir/iterar el el recorrido de todas las hormigas N (pasado en __iteration) veces
            for _ in range(self.__iteration):
                # rutas de las hormigas
                ants_path = []

                # 8. Realizar el recorrido para cada hormiga hormigas
                for _ in range(self.__ants):
                    # crear la matriz de probabilidades para generar los edges (rutas hacia nodos/parámetro)
                    var_proba = AcoTWay.start_matrix_values(self.__parameters, 0.0)

                    # 9. Cada hormiga recorre los parámetros pasando por un ruta (selección de variable) para construir un caso de prueba
                    for pos_param in range(len(self.__parameters)):

                        # var_proba[pos_param] = AcoTWay.explore_probability(pheromone[pos_param], heuristic[pos_param], self.__alpha, self.__beta)
                        # 10. Recorre cada nodo(parámetro), asigna probabilidad a los rutas (variable) por parámetro.
                        #     utiliza el valor de qo para explotar un ruta existente o explorar un nuevo ruta (edge)
                        #     esta sección servirá para eligir el ruta (variable) de mayor probabilidad
                        quu = random.uniform(0, 1)
                        if quu > self.__quu:
                            # exploramos un nuevo ruta
                            var_proba[pos_param] = AcoTWay.explore_probability(pheromone[pos_param], heuristic[pos_param], self.__beta)

                        elif quu <= self.__quu:
                            # explotamos un nuevo ruta
                            var_proba[pos_param] = AcoTWay.exploit_probability(pheromone[pos_param], heuristic[pos_param], self.__alpha, self.__beta)

                    # 11. Guardar la mejor de las rutas las rutas realizada por cada hormiga
                    path = [np.argmax(v) for v in var_proba]
                    ants_path.append(path)

                    # 12. Evaporación de feromona: reducir su presencia por un factor 0<rho<=1. Actualización Local.
                    #     y  iberación de feromonas: las hormigas liberan hormigas en las ruta que ha recorrido (los resumimos)
                    pheromone = AcoTWay.local_pheromones_update(path, pheromone, self.__tau, self.__rho)

                # 13. Seleccionar la mejor ruta (por el número de casos que cubre en las combinaciones tway) realizada por cada hormiga
                best_ant_path = AcoTWay.fitness_function(uncovering_list, covering_list, ants_path)
                candidates_path.append(best_ant_path)  # puede ser vacío, cauando el camino esta cubierto

                # 14. Actualización global de feromonas. Solo se actualiza el mejor
                if best_ant_path[0] != 0:
                    pheromone = AcoTWay.global_pheromones_update(best_ant_path[1], pheromone, self.__rho, best_ant_path[0], len(uncovering_list))

            # 15. Elegir el mejor de todos los candidatos
            best_candidate = AcoTWay.max_candidate(candidates_path)

            # 16. Agregar el mejor candidato a la lista de cobertura total.
            if best_candidate[0] != 0:
                covering_list.append(best_candidate[1])

                # 17. Actualizar la lista de combinaciones no cubiertas.
                for uncov in AcoTWay.cover_uncovering_list(uncovering_list, covering_list, self.__parameters):
                    uncovering_list = tool.remove_array_array(uncovering_list, uncov)

        # 18. Retornar la lista de covertura de los caso de prueba.
        return np.asarray(covering_list)
        """

    # FUNCIONES DE APOYO
    def exploit_ant_path(self, pheromone_values, heuristic_values):
        up = (pheromone_values ** self.__alpha) * (heuristic_values ** self.__beta)
        down = [[u if u > 0 else 1.0] for u in np.sum(up, axis=1)]
        random_exploit = np.asarray([np.asarray(np.random.uniform(0, 1, len(self.__mind_strategy))) for _ in range(len(self.__staff))])
        return random_exploit*(up/down)
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
        return self.generate_staff_task_matrix(np.asarray([self.__tau for _ in range(len(self.__mind_strategy))]))

    def generate_heuristic_values(self):
        total = np.sum(self.__mind_strategy[:, 1])
        return self.generate_staff_task_matrix(np.asarray([1/total for _ in range(len(self.__mind_strategy))]))
        strategy = np.asarray([1/total for _ in range(len(self.__mind_strategy))])

    def generate_staff_task_matrix(self, strategy):
        staff_task_matrix = []
        for i in range(len(self.__task)):
            task_strategy = []
            for j in range(len(self.__staff)):
                task_strategy.append(strategy)
            staff_task_matrix.append(np.asarray(task_strategy))

        return np.asarray(staff_task_matrix)

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
            # # Validando que cada tarea este a cargo de mínimo un empleado
            # if np.min([np.sum(v) for _, v in enumerate(solution_matrix)]) > 0:
            #     # Validando que las personas asignadas a una tarea cubran las habilidades requeridad (skills)
            #     for i in range(len(self.__task)):
            #         a = self.__task_skill[i, 1:]
            #         c = [0 for _ in range(len(a))]
            #         for j, val in enumerate(solution_matrix[i]):
            #             if val > 0:
            #                 b = self.__staff_skill[j, 1:]
            #                 c = [int(c[k] or b[k]) for k in range(len(b))]
            #         d = [int(c[k] and a[k]) for k in range(len(a))]
            #         # print(a, c, d)
            #         if np.sum(a) != np.sum(d):
            #             continue
            #     break

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

        return solution_matrix, np.asarray([project_dur_total,
                                            project_max_dur_total,
                                            project_cost_total,
                                            project_overtime_task_total,
                                            project_overeffort_staff_total,
                                            np.asarray(task_matrix_dur),
                                            np.asarray(tpg_scheduler)])

    def assess_feasibility(self, solution_matrix):
        # Validando que cada tarea este a cargo de mínimo un empleado
        if np.min([np.sum(v) for _, v in enumerate(solution_matrix)]) > 0:
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
        project_overeffort_staff_total = 0
        for i, e in enumerate(self.__staff):
            tpg_scheduler_tmp = tpg_scheduler[:]
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
            # print(i, over_effort)
            project_overeffort_staff_total += 0 if over_effort <= 0 else over_effort

        return project_overtime_task_total, project_overeffort_staff_total

# FIN:FUNCIONES DE APOYO
