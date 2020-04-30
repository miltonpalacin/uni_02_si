import time
import numpy as np
from alglib.meta import aco
from alglib.help import log

# Iniciar el control del tiempo
START_TIME = time.process_time()

set_factor = [["Destacados", "Recientes", "Menor Precio", "Mayor Precio", "Nombre A-Z"],    # 5 posibles valores para el un parámetro
              ["Portabilidad", "Linea Nueva"],                                              # 2 posibles valores para el un parámetro
              ["Aple_SI", "Aple_NO"],                                                       # 2 posibles valores para el un parámetro
              ["Bmovile_SI", "Bmovile_NO"],                                                 # 2 posibles valores para el un parámetro
              ["EKS_SI", "EKS_NO"],                                                         # 2 posibles valores para el un parámetro
              ["Huawey_SI", "Huawey_NO"],                                                   # 2 posibles valores para el un parámetro
              ["LG_SI", "LG_NO"],                                                           # 2 posibles valores para el un parámetro
              ["Motorala_SI", "Motorala_NO"],                                               # 2 posibles valores para el un parámetro
              ["Movistar_SI", "Movistar_NO"],                                               # 2 posibles valores para el un parámetro
              ["Xiaomy_SI", "Xiaomy_NO"],                                                   # 2 posibles valores para el un parámetro
              ["ZTE_SI", "ZTE_NO"],                                                         # 2 posibles valores para el un parámetro
              ["35.9", "45.9", "55.9", "65.9", "75.9", "89.9", "109.9", "149.9"],           # 8 posibles valores para el un parámetro
              ["0-50", "51-150", "151-300", "301-1000", "Más de 1000"]]                     # 5 posibles valores para el un parámetro

# Vector de parámetros con las cantidad de variables (valores posibles) por cada uno (parámetro no uniforme)
VP = [[5, 1], [2, 10], [8, 1], [5, 1]]

# T-Way
T = 2

# Instancia de metaheuristica Ant Colony Optimization (ACO)
np.random.seed()
META_ACO = aco.AcoTWay(VP, T)

# Realizar una prueba ejecutando el método RUN
log.COUNTER_TIME = 10
log.debug_timer("Empezando seguimiento...")
TEST = META_ACO.run()
TEST_NAME = []
for t in TEST:
    TEST_NAME.append([set_factor[t][v] for t,v in enumerate(t)])


# Resultados
print(90*"=")
print(10 * " ", 20*"*", "CASOS DE PRUEBA OPTIMIZADA", 20*"*")
print(90*"=")
print("\n")
print(np.asarray(TEST_NAME))
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
