### Nested Loop

colors = ["red", "blue", "green"]
objects = ["chair", "table", "house"]

for i in colors:
    for j in objects:
        print(i, j)

# multiplication from 1 to 5
for i in range(1, 6):  # 6 is excluded here
    for j in range(1, 11):  # 11 is excluded here
        print(i, "x", j, "=", i * j)

# python functions

# function definition


def add(x, y):  # add is function name, x and y are parameters
    return x + y  # returns the sum of x and y


# calling the function
add(2, 5)  # 2 and 3 are inputs / arguments

add(2, 5, 3)

add(2)


def multiplication(x, y):
    return x * y


multiplication(1, 2)


# arbitraty arguments - we don't define how many arguments we have, it will figure out yourself
def add(*nums):
    sum = 0
    for i in nums:
        sum += i
    return sum


add(1, 2, 3, 4)


# default parameter values
def mul(x, y=3):
    return x * y


mul(4)

# argument ko priority badi huncha define gareko thau vanda
mul(4, 4)


# default arguments should always be defined at the last here, x =3 is default
def substract(y, x=3):
    return x - y


substract(4)

# check keyword argument (homework)

# lambda functions (for small functions, it's a default function, has no names, any number of arguments, only one expression)
# syntax: lambda arguments: expression

sum = lambda x, y, z: x + y + z  # noqa: E731

sum(1, 2, 3)

### Python tuples

# example of tuples = can store multiple values, closed by parenthesis, immutable, allows duplicate values, ordered

# difference between list and tuple is list is mutable and tuple is immutable

# all the same operations of list can be done in tuple

# multiply tuples

(1, 2, 3) * 3

# tuples value cannot be changed, immutable
test = (1, 2, 3)
test[0] = 5

# python sets
# sets are unordered, unindexed, doesn't allow duplicates, immutable, {}, not used extensively

# duplicates aren't retained
set_1 = {1, 2, "hi", 3, "hi"}
set_1

# python dictionary = key and values
