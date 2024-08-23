# python dictionary

city_pop = {"Chicago": 3.5, "New York": 7, "LA": 6}

city_pop[0]

city_pop["Chicago"]

# should have keys for indexing, not 0,1,2 here "chicago" is used to find it's value
# it's mutable, and ordered

city_pop.get("LA")

city_pop.keys()

city_pop.values()

city_pop.items()

"Chicago" in city_pop

"San Francisco" in city_pop

# dict[key]= value

city_pop["LA"] = 20

city_pop.update({"Chicago": 30})

# add new items in dict, don't have duplicate keys though

city_pop["Kathmandu"] = 2.5

# remove items
city_pop.pop("Chicago")

city_pop.clear()

# clear, empties the dictionary
# delete chai dictionary lai nei delete gardincha

del city_pop

# OOP concepts

# object priority high? then mathi run vairako code lai chai run nagari sabai run garcha
# error vaye pani run huncha tala ko

# class and objects are imp

# class has it's own properties, we need to define it using objects.

# class has data and functions, class chalauna object chaicha

# everything is an object in python

# int, str, list, tuple, dict everything is an object

# class is like a blueprint for building objects

# "copies" of the blueprint are called object instances


class Student:  # class ko first letter should always be capital
    pass  # pass means to do nothing


student1 = Student()  # instance of the class Student
student2 = Student()  # can have multiple objects of the same class
print(type(student1))

id(student1)

id(student2)

# data in class can be defined as variable


class Student:
    # class variable
    school = "School of Data Science"


student1 = Student()
student2 = Student()
print(student1.school)
print(student2.school)


class Student:
    # constructor is a function
    def __init__(self, name, level):  # self vaneko yei class bhitra garirachau vanera
        # instance variables
        self.name = name
        self.level = level


# self is reserved

student1 = Student("Salish Shakya", 1)
student2 = Student("Silex Shakya", 2)

student1.name
student2.level

# __init__ and __str__ default constructors are called auto, but ones that we define need to be called separately


class Student:
    def __init__(self, name, level):  # self vaneko yei class bhitra garirachau vanera
        # instance variables
        self.name = name
        self.level = level

    # regular constructor
    def level_up(self):
        self.level += 1


sam = Student("Sam Smith", 1)
print(sam)


class Student:
    def __init__(self, name, level):  # self vaneko yei class bhitra garirachau vanera
        # instance variables
        self.name = name
        self.level = level

    # regular constructor
    def level_up(self):
        self.level += 1

    def __str__(self):
        return "{} studies at level {}".format(self.name, self.level)


# output ali hawa airacha so mathi ko gareko

print(sam)


class Student:
    # constructor
    def __init__(self, name, level):
        self.name = name
        self.level = level

    def __str__(self):
        return "{} studies at level {}".format(self.name, self.level)

    # regular method
    def level_up(self):
        self.level += 1


sam = Student("Sam Smith", 1)
tom = Student("Tom Cruise", 3)
print(sam)
print(tom)

sam.level_up()
