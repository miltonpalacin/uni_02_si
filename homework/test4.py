from alglib.meta import aco


# Vector de cantidad de variables por cantidad de parámetros
VP = [[2, 2]]
META_ACO = aco.AcoTWay(VP, 1)
TEST = META_ACO.run()
print(TEST)
