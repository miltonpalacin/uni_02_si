import numpy as np
#from datetime import datetime
#from alglib.help import log, config

#from alglib.meta import aco
"""
print(datetime.now().strftime('%d/%m/%Y %H:%M:%S.%f'))
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])


config.SETTING.DEBUG_ENABLE = True
config.SETTING.TRACE_ENABLE = True

log.line("milton")
log.warning("warning")
log.debug("debgg")
log.trace("trace")
log.line_csv()
log.line_csv("12", "asdasd")
"""
"""
parts = [str(x) for x in [1,2]]
print(" ".join(parts))
"""

#aco.test()
"""
def test(variable):
    give = []
    for i in variable:
        give.append(list(range(i)))
    return give

print([0 for _ in range(3) ])
v = test([2,3]);
print(v[1][2])
v[1][2] = 5
print(v)678+9555
print(np.asarray(v))
print(np.asarray(v)[1:1:2])
"""
import collections
a = [1,1,1,1,2,2,2,2,3,3,4,5,5]
v = np.asarray(a)
z = [1,2,3,4]
z2 = [1,2,3,4]
give = []
give.append(np.asarray([0.1*i for i in z]))
give.append(np.asarray([0.3 for _ in z2]))
give.append(np.asarray([0 for _ in z]))

give[0][1] = 145
print(give[0][1])
print(give)

"""
for i in v:
    print(i)

from itertools import groupby
v = [len(list(group)) for key, group in groupby(a)]
print(v)

var_count = collections.Counter(a)
print(var_count)
var_max = max(var_count.values())
var_min = min(var_count.values())
print(var_max)
print(var_min)
"""

=================================================
import random
import numpy as np
from alglib.meta.aco import AcoTWay


def pheromone_matrix(parameters, tau):
    give = []
    for i in parameters:
        give.append(np.asarray([tau for _ in range(i)]))
    return np.asarray(give)

g = AcoTWay.initial_unconvering_list(np.asarray([2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 1]), 2)
print(g)
"""
f = AcoTWay.start_matrix_values([2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 1], .5)
h = AcoTWay.compute_heuristic_values([2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 1], [])
print(f)
print(h)
print("==========================================================")
print ((.5**23)*f)
print("==========================================================")
s = (f[21]**0.5)/(h[21]**3)
t = np.sum(s)
#print(s, t)
f[21] = s/t
print(f[21])
#print(f)
print("==========================================================")

print("==========================================================")
c = AcoTWay.exploit_probability(f, h, .3, 3)
c[0][0] = 0.3
c[0][1] = 0.7
c[21][5] = 0.8
print(np.asarray(c))
print("==========================================================")
arr = []
x =[np.argmax(v) for v in c]
arr.append(x)
arr.append(x)
print(np.asarray(arr))
print("==========================================================")
# for i in range(np.asarray([2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,6,1]).size):
#    print(i)
"""
"""
a = pheromone_matrix(np.asarray([1,2,3,4]), 0.5)
b = pheromone_matrix(np.asarray([1,2,3,4]), 2)
c = (a**.5)*(b**.7)
print("==========================================================")
print(a)
print(b)
print(c)
print("==========================================================")
d = np.asarray([np.sum(v) for v in c])
print(c/d)
print("==========================================================")

con_a = []
if not con_a:
    give = []
    for i in [2, 2]:
        give.append(np.asarray([j for j in range(i)]))
    con_a = np.asarray(give)
print(con_a)
print(AcoTWay.compute_heuristic_values([2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1], []))
"""
# print("------------")
# print(a**0.7/a)
# print("------------")
##print(np.ones((2, 2))**2)
# print("------------")
# print(np.array(a))
# print("------------")

# print(list(range(3)))
# print([*range(3)])
# print(np.arange(3))
==========================================================================
import numpy as np

v = np.empty((0, 3), int)

v = np.append(v, [[1, 2, 3]], axis=0)
v = np.append(v,[[7,2,3]],axis=0)
print(v)

arr = []
arr.append([1,2,3])
arr.append([4,5,6])
print(np.asarray(arr))

to = [[1,2,3],[0,2,3],[1,1,3],[0,2,0],[0,2,1],[0,2,2]]
ind = [0,1]
dd = []
cp  = np.asarray([1.0,3])
print("---------")
#dd.append(cp)
dd.append(ind)
print(dd)

x = [0,2,3]
check = np.unique(np.asarray(to)[:,ind],axis=0) 
print("---------")
print(check)
print("---------")
print(np.asarray(x)[ind])
print("---------")
print((check == cp).all(1).any())
print("---------")
print(int((np.asarray(to) == x).all(1).any()))


# if (np.unique(np.asarray(to)[:,ind] == b).all():
#     print("coincide")
# else:
#     print("NOO")

===================================================================
import numpy as np
from alglib.help import combination
from alglib.meta.aco import AcoTWay
from alglib.help import tool

a = combination.bivector_to_vector([[2, 3], [4, 3], [3, 1]])
b = AcoTWay.start_matrix_values(a, .5)
print(b)
v = AcoTWay.local_pheromones_update([1, 0, 1, 3, 2, 1, 0], b, 0.4, 0.5)
print(v)
"""
print (b)

print(list(range(len(b))))
print(len(b[3]))
print(a)
for i,item in enumerate(a):
    print("{} : {}".format(i,item))





b = AcoTWay.initial_unconvering_list(a, 3)
c = [0, 1, 2]
d = [4, 5, 6]
# print(b)
x = 0
for i in b:

    if(i == d).all():
        break
    x += 1
#b = np.delete(b, x, axis=0)
b = tool.remove_array_array(b,d)
print(x, b)

#v = np.argwhere(b == [d])
#print(v)
"""
"""
c = [4,5,6]
print(a)
# print(b)
sol = [1,2,3,4]
d =[]
d.append(np.asarray([2,sol, []]))
d.append(np.asarray([3,sol, []]))
d.append(np.asarray([5,sol, []]))
d.append(np.asarray([2,sol, []]))
print(d)
print(np.argmax(np.asarray(d)[:,0]))
print(d[np.argmax(np.asarray(d)[:,0])])
x = d[np.argmax(np.asarray(d)[:,0])]
x[2] = [4,5]
# np.insert(x, [[123,123123], [12,12]], axis =0)
print(x[0])
print(len(d))
"""
=============================================================
from alglib.meta import aco
import time


# Vector de cantidad de variables por cantidad de parámetros
start = time.process_time()

#VP = [[2, 4]]
#VP = [[2, 10]]
#VP = [[5, 1],[3,8],[2, 2]]
#VP = [[2, 2], [1, 1], [2, 8], [2, 3], [2, 1], [2, 6], [2, 1], [1, 1]]
"""
-- Tipo contenido 12 
-- Discioplia 5
-- Subdisciplina 5
-- lenguanke 5
-- Manuevo mas viejo 2
-- Textot 1
"""
#VP = [[12, 1], [5, 3],[2,1], [1,1]]
VP = [[5, 1], [2, 10], [8, 1], [5, 1]]

#VP = [[5, 2],[4,2],[3, 2]]
T = 2
META_ACO = aco.AcoTWay(VP, T)
TEST = META_ACO.run()
print(TEST)
print(len(TEST))

end = time.process_time()
print("Elapsed Time: " + str(end - start))
==================================================================
############
### CASO DE PRUEBA 01
###
###########


q = Queue()
runs = []
 think you can just do for runs = [run for run in iter(lambda: q.get(False),None)]