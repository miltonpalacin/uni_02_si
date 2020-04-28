from alglib.meta import aco
import time


# Vector de cantidad de variables por cantidad de parámetros
start = time.process_time()

#VP = [[2, 10]]
#VP = [[5, 1],[3,8],[2, 2]]
#VP = [[5, 2],[4,2],[3, 2]]
VP = [[2, 2], [1, 1], [2, 8], [2, 3], [2, 1], [2, 6], [2, 1], [1, 1]]
META_ACO = aco.AcoTWay(VP, 4)
TEST = META_ACO.run()
print(TEST)
print(len(TEST))

end = time.process_time()
print("Elapsed Time: " + str(end - start))