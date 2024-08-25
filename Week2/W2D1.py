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

arr2d2 = np.array([[7, 8, 9], [9, 5, 6]])

np.hstack([arr2d, arr2d2])

x = [1, 2, 3, 4, 5, 6, 7, 8]
x1, x2, x3 = np.vsplit(x, [3, 5])
print(x1, x2, x3)

# vsplit and hsplit requires 2D arrays hence the code will show error

x = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])  # this is a 2d array
x1, x2, x3 = np.vsplit(x, [3, 5])
print(x1, x2, x3)

# 2D array with 2 rows and 8 columns
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]])

# Split along columns at indices 3 and 5
x1, x2, x3 = np.hsplit(x, [3, 5])

print("x1:", x1)
print("x2:", x2)
print("x3:", x3)

# 2D array with 6 rows and 4 columns
x = np.array(
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20],
        [21, 22, 23, 24],
    ]
)

# Split along rows at indices 2 and 4
x1, x2, x3 = np.vsplit(x, [2, 4])

print("x1:", x1)
print("x2:", x2)
print("x3:", x3)

####hsplit is for splitting along columns, so it’s most useful for arrays with multiple columns.
####vsplit is for splitting along rows, so it’s most useful for arrays with multiple rows.

##use of timeit


# Quick function: calculates the sum of a list
def quick_function():
    return sum(range(10000))


# Slow function: performs a nested loop with a large number of iterations
def slow_function():
    result = 0
    for i in range(1000):
        for j in range(1000):
            result += i * j
    return result


# Using %timeit to measure execution time
# %timeit quick_function()
# %timeit slow_function()

# The %timeit magic command is specifically designed for Jupyter notebooks and is not valid in regular Python scripts (.py files). In .py files, you should use the timeit module instead of the %timeit magic command.

# 112 μs ± 8.52 μs per loop here, microseconds (µs).
# 56.8 ms ± 2.22 ms per loop here, milliseconds (ms).

# The quick function runs in microseconds (µs).
# The slow function runs in milliseconds (ms).

# check the execution time of numpy array and normal python list using %timeit

# Create a large Python list and a NumPy array
python_list = list(range(1000000))
numpy_array = np.arange(1000000)

# Measure the time taken for a simple operation (like summing all elements)
# print("Timing for Python list:")
# # %timeit sum(python_list)
print(sum(python_list))

# print("\nTiming for NumPy array:")
# # %timeit np.sum(numpy_array)
print(np.sum(numpy_array))

# NumPy is usually faster for numerical operations due to its underlying C implementation and optimizations
