# coding: UTF-8

import random
import collections
import itertools as it
import numpy as np
from ..help import combination


class AcoTWay:

    def __init__(self, parameters_variables=None, t_way=None):
        """
        Implementación de Ant Colony Optimization para generación de casos de prueba T-Way (de t formas):
        Se considera un generación de casos de prueba con todas la interacciones posibles.
        Se puede considerar como un camino para que las hormigas se muevan, como se muestra en la Figura 1.
        La Figura 1 muestra el espacio de búsqueda de un algoritmo de generación de prueba, uno a la vez.
        El recorrido de una hormiga debe ser tal que cada caso de prueba no cubierto deberá ser explorado
        """
        # cada valor del parámetro representa la cantidad de variables que puede aceptar o cambiar.
        self.__parameters = combination.bivector_to_vector(parameters_variables)
        self.__t_way = t_way
        self.config(ants=20, alfa=0.5, beta=3, rho=0.5, tau=0.4, quu=0.5, iteration=5)

    def config(self, ants=None, alfa=None, beta=None, rho=None, tau=None, quu=None, iteration=None):
        self.__ants = ants              # número de hormigas
        self.__alfa = alfa              # coeficiente para el control de la influencia/peso de la cantidad de feromonas
        self.__beta = beta              # coeficiente para el control de la influencia/peso de la inversa (una ruta/distancia) de la distancia
        self.__rho = rho                # tasa de volatilidad de las feromonas
        self.__tau = tau                # valor inicial de la feromona
        self.__quu = quu                # Valor que permite la explotación o exploración de nuevos caminos (regla de proporcionalidad aleatoria) para
        self.__iteration = iteration    # máximo de iteraciones (númoer de veces) que hará el recorrido de todas hormiga

    def run(self):
        # 1. Generar todas las posibles interacciones basado en el total de valores que pueden tomar cada uno de los parámetros
        #    V1^P1, V2^P2, V3^P3,..., Vn^Pn
        uncovering_list = AcoTWay.initial_unconvering_list(self.__parameters, self.__t_way)

        # 2. Iniciar la matriz de covertura
        covering_list = np.asarray([])

        # 3. Iniciar la principales variables
        #    el número de iteración y el número de hormigas se han asignado en el constructor

        # 4. Recorrer todos los casos de prueba generados  posibles
        while uncovering_list.size > 0:

            # 5. Iniciar matriz de feromonas con la constante ingresada _tau
            pheronome = AcoTWay.start_matrix_values(self.__parameters, self.__tau)

            # 6. Cálculo inicial de la heuristica para cada recorrido de la hormiga
            heuristic = AcoTWay.compute_heuristic_values(self.__parameters, covering_list)

            # 7. Repetir/iterar el el recorrido de todas las hormigas N (pasado en __iteration) veces
            for _ in range(self.__iteration):

                # 8. Realizar el recorrido para cada hormiga hormigas
                for _ in range(self.__ants):
                    # crear la matriz de probabilidades para generar los edges (caminos hacia nodos/parámetro)
                    var_proba = AcoTWay.start_matrix_values(self.__parameters, 0)
                    
                    # 9. Cada hormiga recorre los parámetros pasando por un camino(variable) para construir un caso de prueba
                    for pos_param in range(self.__parameters.size):

                        # 10. Recorre cada nodo(parámetro), asigna probabilidade a las selección del camino (variable) por parámetro.
                        #     utiliza el valor de qo para explotar un camino existente o explorar un nuevo camino (edge)
                        quu = random.uniform(0, 1)
                        if quu > self.__quu:
                            # exploramos un nuevo camino
                            var_proba[pos_param] = AcoTWay.explore_probability(pheronome[pos_param], heuristic[pos_param], self.__alfa, self.__beta)

                        elif quu <= self.__quu:
                            # explotamos un nuevo camino
                            var_proba[pos_param] = AcoTWay.exploit_probability(pheronome[pos_param], heuristic[pos_param], self.__alfa, self.__beta)

                # 11
    # FUNCIONES DE APOYO

    @staticmethod
    def compute_heuristic_values(parameters, convering_list):
        give = []
        if convering_list:
            parameter = 0
            for i in parameters:
                values = range(i)
                give.append(np.asarray([0 for _ in values]))

                # contar numero de veces en CA_set optimo
                val_count = collections.Counter(convering_list[:, parameter])
                val_max = max(val_count.values())
                val_min = min(val_count.values())
                # calculando el valor heurístico nij (inverse de la distancia)
                for k in values:
                    give[parameter][k] = (val_max - val_count[k] - 1)/(val_max - val_min - 1)
                parameter += 1
        else:
            for i in parameters:
                values = range(i)
                give.append(np.asarray([1 for _ in values]))

        return np.asarray(give)

    @staticmethod
    def start_matrix_values(parameters, tau):
        give = []
        for i in parameters:
            give.append(np.asarray([tau for _ in range(i)]))
        return np.asarray(give)

    @staticmethod
    def explore_probability(pheromone_values, heuristic_values, alpha, beta):
        top = (pheromone_values**alpha) * (heuristic_values ** beta)
        bottom = np.asarray([np.sum(v) for v in top])
        return top/bottom
    
    @staticmethod
    def exploit_probability(pheromone_values, heuristic_values, alpha, beta):
        # diminuir la influencia de la heuristica
        top = (pheromone_values**alpha) / (heuristic_values ** beta)
        bottom = np.asarray([np.sum(v) for v in top])
        return top/bottom

    @staticmethod
    def initial_unconvering_list(parameters, t_way):
        return np.asarray(list(it.combinations(np.arange(parameters.size), t_way)))

    # FIN:FUNCIONES DE APOYO
