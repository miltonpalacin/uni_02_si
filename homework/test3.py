import numpy as np
from alglib.help import combination
from alglib.meta.aco import AcoTWay

a = combination.bivector_to_vector([[2, 3], [4, 3], [3, 1]])
#b = AcoTWay.initial_unconvering_list(a, 3)
c = [4,5,6]
print(a)
#print(b)
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
#np.insert(x, [[123,123123], [12,12]], axis =0)
print(x[0])
print(len(d))