from Fibo import fib
import Fibo as fibo
import num
# module directory bhitra gayera khojcha

fibo.fib(100)
num.sub(100, 1)
num.sum(100, 1)

dir(fib)
# dir le yo file bhitra k k cha vanera

# first library that we use is numpy
# numpy for numericals, pandas for visualization, matplot, seaborn for graph

# install numpy, pandas, matplotlib using pip install

# numpy alias is np
import numpy as np  # noqa: E402

np.__version__

# numpy is np, pandas is pd, matplot is plt

# numpy arrays are list but are more efficient storage and operations on data than python lists

my_arr = np.array([2, 4, 8, 16])

# numpy ko input sabai array mei kaam garcha not in list

type(my_arr)

np.array([1, 2.4, 3, 4, 5])

# everything on float, only takes one data type and changes type forcefully to the more storage one

np.array(["string", 1, 2.0])

# dtype='<U32' string type is universal 32 bit

np.array([1, 2, 3, 3.5], dtype="int16")

# forcefully int ma lageko with dtype

# multi-dimensional array

np.array(
    [
        # 3 by 3 array
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
)

np.zeros(5, dtype="int")

# here x,y,z huncha
np.ones((3, 5, 2), dtype=float)

np.arange(1, 31, 3)

# python ko range, numpy ko arange

# linear spacing
np.linspace(0, 1, 5)

# numpy standard data types check in notes
