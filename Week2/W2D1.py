import numpy as np

x1 = np.random.randint(10, size=6)
x2 = np.random.randint(10, size=(4, 3))
x3 = np.random.randint(10, size=(2, 3, 4))
x1

# ndim = no. of dimensions?

print(x1.ndim)

print(x2.ndim)

print(x3.ndim)

# shape = size of each dimension (v .imp)

print(x1.shape)
print(x2.shape)
print(x3.shape)

# size = how many elements in total

print(x1.size)
print(x2.size)
print(x3.size)

# shape and size is the first step in DS mostly

x1.dtype

# x1 is an array which has int. x1 associated with numpy

print("x2.itemsize =", x2.itemsize)
print("x2.nbytes=", x2.nbytes)

print(x1)
print(x1[0])

print(x2)

print("item at 3rd row and 2nd column is", x2[2, 1])

print("item at 3rd row and last column is", x2[2, -1])

x2[2, -1] = 3.1415
print(x2)
print("Updated = ", x2[2, -1])

x = np.arange(10)

# slicing

x[:7]
x[3:8]
x[::2]
x[2::2]
x[7:3:-1]  # index 7 to index 4 decrement by 1

x[::]

x2

x2[:3, :2]  # rows = index 0 to 2, columns = index 0 to 1

x2[::2, :]  # every other 2 rows, all columns

x2[::-1, ::-1]

x2[:, 1]

# reshaping arrays

z = np.arange(1, 10)
z.reshape((3, 3))

z.reshape((4, 4))

# array concatenation and spliting

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
np.concatenate([x, y])

arr2d = np.array([[1, 2, 3], [4, 5, 6]])
np.concatenate([arr2d, arr2d])  # by default rpws ma huncha i.e. x=0

arr2d = np.array([[1, 2, 3], [4, 5, 6]])
np.concatenate([arr2d, arr2d], axis=1)  # axis 1 vaneko columns

# horizontal and vertical add is hstack and vstack

np.vstack([x, arr2d])

# do hstack yourself at home

# splitting arrays

x = [1, 2, 3, 4, 5, 6, 7, 8]
x1, x2, x3 = np.split(x, [3, 5])
print(x1, x2, x3)

# do hsplit and vsplit at home

# check how much time it took for operation %timeit check in pdf at home

x = np.array([-2, -3, 10, -1])
np.abs(x)

theta = np.linspace(0, np.pi, 3)

theta

print("sin(theta) =", np.sin(theta))

# linspace is linear spacing so pi is 180 degree so 0 to 180 ko three spaces lincha
