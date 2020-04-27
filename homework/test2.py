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
