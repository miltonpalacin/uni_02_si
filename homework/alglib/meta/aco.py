# coding: UTF-8

import collections
import itertools as it
import numpy as np
from ..help import combination, tool


class AcoTWay:

    def __init__(self, parameters_variables=None, t_way=None):
        """
        Implementación de Ant Colony Optimization para generación de casos de prueba T-Way (de t formas):
        Se considera un generación de casos de prueba con todas la interacciones posibles.
        Se puede considerar como un ruta para que las hormigas se muevan, como se muestra en la Figura 1.
        La Figura 1 muestra el espacio de búsqueda de un algoritmo de generación de prueba, uno a la vez.
        El recorrido de una hormiga debe ser tal que cada caso de prueba no cubierto deberá ser explorado
        """
        # cada valor del parámetro representa la cantidad de variables que puede aceptar o cambiar.
        self.__parameters = combination.bivector_to_vector(parameters_variables)
        self.__t_way = t_way
        self.config(ants=5, alfa=0.5, beta=3, rho=0.5, tau=0.4, iteration=2)

    def config(self, ants=None, alfa=None, beta=None, rho=None, tau=None, iteration=None):
        self.__ants = ants              # número de hormigas
        self.__alfa = alfa              # coeficiente para el control de la influencia/peso de la cantidad de feromonas
        self.__beta = beta              # coeficiente para el control de la influencia/peso de la inversa (una ruta/distancia) de la distancia
        self.__rho = rho                # tasa de volatilidad de las feromonas
        self.__tau = tau                # valor inicial de la feromona
        self.__iteration = iteration    # máximo de iteraciones (númoer de veces) que hará el recorrido de todas hormiga

    def run(self):
        # 1. Generar todas las posibles interacciones basado en el total de valores que pueden tomar cada uno de los parámetros
        #    V1^P1, V2^P2, V3^P3,..., Vn^Pn
        uncovering_list = AcoTWay.initial_uncovering_list(self.__parameters, self.__t_way)

        # 2. Iniciar la matriz de covertura
        covering_list = []

        # 3. Iniciar la principales variables
        #    el número de iteración y el número de hormigas se han asignado en el constructor

        # 4. Recorrer todos los casos de prueba generados  posibles
        while len(uncovering_list) > 0:
            # rutas candidatas
            candidates_path = []

            # 5. Iniciar matriz de feromonas con la constante ingresada _tau
            pheromone = AcoTWay.start_matrix_values(self.__parameters, self.__tau)

            # 6. Cálculo inicial de la heuristica para cada recorrido de la hormiga
            heuristic = AcoTWay.compute_heuristic_values(self.__parameters, covering_list)

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

                        var_proba[pos_param] = AcoTWay.explore_probability(pheromone[pos_param], heuristic[pos_param], self.__alfa, self.__beta)

                    # 11. Guardar la mejor de las rutas las rutas realizada por cada hormiga
                    path = [np.argmax(v) for v in var_proba]
                    ants_path.append(path)

                    # 12. Evaporación de feromona: reducir su presencia por un factor 0<rho<=1. Actualización Local.
                    #     y  iberación de feromonas: las hormigas liberan hormigas en las ruta que ha recorrido (los resumimos)
                    pheromone = AcoTWay.local_pheromones_update(path, pheromone, self.__tau, self.__rho)

                # 13. Seleccionar la mejor ruta (por el número de casos que cubre en las combinaciones tway) realizada por cada hormiga
                best_ant_path = AcoTWay.fitness_function(uncovering_list, covering_list, ants_path, self.__parameters)
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
                for uncov in best_candidate[2]:
                    uncovering_list = tool.remove_array_array(uncovering_list, uncov)

        # 18. Retornar la lista de covertura de los caso de prueba.
        return np.asarray(covering_list)

    # FUNCIONES DE APOYO

    @staticmethod
    def compute_heuristic_values(parameters, covering_list):
        give = []

        if covering_list:
            covering = np.asarray(covering_list)
            for idx, val in enumerate(parameters):
                values = range(val)
                # genera entrada para los valores de las variables
                give.append(np.asarray([0.0 for _ in values]))

                # contar numero de veces en CA_set optimo
                val_count = collections.Counter(covering[:, idx])
                val_max = max(val_count.values())
                val_min = 0
                # Si no se cuenta algúna varaible el menor valor es 0
                if len(val_count) == val:
                    val_min = min(val_count.values())
                # calculando el valor heurístico nij (inverse de la distancia)
                for k in values:
                    give[idx][k] = (val_max - val_count[k] + 1)/(val_max - val_min + 1)
        else:
            for i in parameters:
                give.append(np.asarray([1.0 for _ in range(i)]))

        return np.asarray(give)

    @staticmethod
    def start_matrix_values(parameters, val):
        give = []
        for i in parameters:
            give.append(np.asarray([val for _ in range(i)]))
        return np.asarray(give)

    @staticmethod
    def explore_probability(pheromone_values, heuristic_values, alpha, beta):
        top = (pheromone_values**alpha) * (heuristic_values ** beta)
        bottom = np.sum(top)
        quu = np.random.uniform(0, 1, len(pheromone_values))  # random.uniform(0, 1)
        return quu * (top/bottom)

    @staticmethod
    def exploit_probability(pheromone_values, heuristic_values, alpha, beta):
        top = (pheromone_values**alpha) * (heuristic_values ** beta)
        bottom = np.sum(top)
        return top / bottom

    @staticmethod
    def initial_uncovering_list(parameters, t_way):
        return np.asarray(list(it.combinations(np.arange(len(parameters)), t_way)))

    @staticmethod
    def fitness_function(uncovering_list, covering_list, cases_test, parameters):
        give = []
        max_case = 0
        max_rem = 0
        if covering_list:
            cover = np.asarray(covering_list)
            for test in cases_test:

                if not (cover == test).all(1).any():
                    rem = []  # para remover los casos que se lograron cubrir
                    count = 0
                    for uncov in uncovering_list:
                        # extraer todas la combinaciónes ya realizadas en la matriz de cobertura
                        check = np.unique(cover[:, uncov], axis=0)
                        # extraer la combinación del caso de prueba y verificar si cubre una combinación adicional
                        case = np.asarray(test)[uncov]
                        if(check == case).all(1).any():
                            continue

                        # total de casos de prueba que deben existir con esta combinación t-way
                        tot = 1
                        for i in uncov:
                            tot = tot * parameters[i]

                        # verificar si con el caso adicional se cubre el la combinación t-way puntual
                        if tot == (len(check)+1):
                            rem.append(uncov)

                        count += 1

                    # validar también que el mejor caso sea el que pueda elimnar más combinaciones en caso complete varias
                    if max_case <= count and count != 0:
                        if max_case == count:
                            if max_rem <= len(rem):
                                max_rem = len(rem)
                                give = [count, test, rem]
                        else:
                            max_case = count
                            give = [count, test, rem]

            if max_case == 0:
                give = [0, [], []]
        else:
            count = len(uncovering_list)
            give = [count, cases_test[0], []]

        return np.asarray(give)

    @staticmethod
    def cover_uncovering_list(uncovering_list, covering_list, parameters):
        give = []
        if covering_list:
            cover = np.asarray(covering_list)
            for uncov in uncovering_list:

                # extraer todas la combinaciónes ya realizadas en la matriz de cobertura
                check = np.unique(cover[:, uncov], axis=0)

                # total de casos de prueba que deben existir con esta combinación t-way
                tot = 1
                for i in uncov:
                    tot = tot * parameters[i]

                # verificar si con el caso adicional se cubre el la combinación t-way puntual
                if tot == len(check):
                    give.append(uncov)
        else:
            give = []

        return np.asarray(give)

    @staticmethod
    def max_candidate(candidates_path):
        max_cand = 0
        best_cand = []
        max_rem = 0

        for candidate in candidates_path:
            val = candidate[0]

            # validar también que el mejor caso sea el que pueda elimnar más combinaciones
            if max_cand <= val and val != 0:
                if max_cand == val:
                    if max_rem <= len(candidate[2]):
                        max_rem = len(candidate[2])
                        best_cand = candidate
                else:
                    max_cand = val
                    best_cand = candidate

        if max_cand == 0:
            best_cand = [0, [], []]

        return best_cand

    @staticmethod
    def local_pheromones_update(path, pheromone, tau, rho):
        for idx, edge in enumerate(path):
            for var in range(len(pheromone[idx])):
                if var != edge:
                    pheromone[idx][var] = rho * pheromone[idx][var]
                else:
                    pheromone[idx][var] = (1 - rho) * pheromone[idx][var] + rho*tau
        return pheromone

    @staticmethod
    def global_pheromones_update(path, pheromone, rho, fi_best, tot_uncover):
        for idx, edge in enumerate(path):
            for var in range(len(pheromone[idx])):
                if var == edge:
                    pheromone[idx][var] = (1 - rho) * pheromone[idx][var] + rho*(fi_best/tot_uncover)
        return pheromone

    # FIN:FUNCIONES DE APOYO
