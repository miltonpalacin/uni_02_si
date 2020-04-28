from alglib.meta import aco
import time


# Vector de cantidad de variables por cantidad de parámetros
start = time.process_time()

VP = [[3, 5]]
META_ACO = aco.AcoTWay(VP, 2)
TEST = META_ACO.run()
print(TEST)

end = time.process_time()
print("Elapsed Time: " + str(end - start))