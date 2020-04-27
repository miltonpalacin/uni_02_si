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
