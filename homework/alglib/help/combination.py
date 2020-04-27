# coding: UTF-8

import itertools as it
import numpy as np


def bivector_to_vector(input_array):
    give = []
    for i in input_array:
        for _ in range(i[1]):
            give.append(i[0])
    return np.asarray(give)


def create_full_array(input_array):
    """
    Función que crea todas las posibles combinaciones con cada valor posible de los elementos del array.
    """
    give = []
    for i in input_array:
        give.append(np.asarray(np.arange(i)))

    arr = []
    # Utilizar el producto cartersiano de conjuntos
    for i in it.product(*give):
        arr.append(list(i))

    return np.asarray(arr)


import collections
#print(bivector_to_vector([[2, 2],[3,7]]).size)
#print(bivector_to_vector([[2, 2], [1, 1], [2, 8], [2, 3], [5, 1], [3, 6], [2, 1], [1, 1]]))
#print(create_full_array([2, 2]))
"""
p = bivector_to_vector([[2, 2], [2, 3], [3, 8]])
print(p)
f = [p,p]
v = create_full_array(p)
#print(f[0, [1,2]])

#v[:,(0,1)]

# (check == full[i, p]).all(1)
curr = np.asarray([v[123, :]])
#print(curr," ",curr[:,(0,1)], " ", np.unique(curr[:, (10,12)], axis=0)," ", [w for w in it.combinations(np.arange(v.shape[1]), 2)][1])
#print(np.asarray([np.asarray(w) for w in it.combinations(np.arange(v.shape[1]),2)]))
print(np.asarray(list(it.combinations(np.arange(v.shape[1]),2)))**2)
"""
#print(create_full_array(bivector_to_vector([[2, 2], [2, 3], [3, 8]])))
cc = collections.Counter(create_full_array(bivector_to_vector([[2, 2], [2, 3], [3, 8]]))[:,0])
print(cc,cc[0], cc[1], cc[2], cc[3])
# print(create_full_array([2,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1]).shape)
# print(create_full_array([[2, 2], [1, 1], [2, 8], [2, 3], [2, 1], [2, 6], [2, 1], [1, 1]]).shape)
# print(list(np.arange(2)))
