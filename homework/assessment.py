import time
import numpy as np
from alglib.meta import aco
from alglib.help import log



# Vector de parámetros con la cantidad de variables por cada parámetros

# Iniciar el control del tiempo
START_TIME = time.process_time()

#VP = [[10, 10]]
#VP = [[2, 4]]
#VP = [[2, 10]]
#VP = [[5, 1],[3,8],[2, 2]]
#VP = [[2, 2], [1, 1], [2, 8], [2, 3], [2, 1], [2, 6], [2, 1], [1, 1]]
#VP = [[5, 2],[4,2],[3, 2]]
#VP = [[12, 1], [5, 3],[2,1], [1,1]]
#VP = [[5, 1], [3, 8],[2,2]]
#VP = [[5, 2], [4, 2],[3,2]]
#VP = [[3, 13]]
"""
-- Tipo contenido 12 
-- Discioplia 5
-- Subdisciplina 5
-- lenguanke 5
-- Manuevo mas viejo 2
-- Textot 1
"""
# Caso de Eduardo Yauri
VP = [[5, 1], [2, 10], [8, 1], [5, 1]]

# T-Way
T = 2

# Instancia de metaheuristica Ant Colony Optimization (ACO)
META_ACO = aco.AcoTWay(VP, T)

# Realizar una prueba ejecutando el método RUN
log.COUNTER_TIME = 10
log.debug_timer("Empezando seguimiento...")
TEST = META_ACO.run()

# Resultados
print(90*"=")
print(10 * " ", 20*"*", "CASOS DE PRUEBA OPTIMIZADA", 20*"*")
print(90*"=")
print("\n")
print(TEST)
print("\n")
print(90*"=")
print(10 * " ", 20*"*", "RESULTADOS", 20*"*")
print(90*"=")
print("\n")
print("Total de pruebas exhaustiva:", 4*"\t", '{:10,.2f}'.format(np.prod([pow(a[0], a[1]) for a in VP])))
print("Todal de casos de pruebas optimizada a ", T, "- way:", 0*"\t", len(TEST))
print("Porcentaje optimizado a ", T, "- way:", 4*"\t", (1 - len(TEST)/np.prod([pow(a[0], a[1]) for a in VP])))

# Finalizar el control del tiempo
END_TIME = time.process_time()
print("Tiempo utilizado: ", 7*"\t", str(END_TIME - START_TIME))
