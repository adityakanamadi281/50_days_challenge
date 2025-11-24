# OOPs in python 


# To map with real world scenarios, we started using objects in code 
# this is called object oriented programming.


# procedural programming is a type of programming where we write  code in line by line.
# Functins are used to perform a specific task.

# Object oriented programming is a type of programming where we write code in a way that is easy to understand and easy to maintain.




# Class and Object in python

# class is a bblueprint for creating objects.
# object is instance of class.



# class is a collection of objects.




class Student:
    name="Aditya"
    age=22
    city="Banglore"

s1=Student()
print(s1.name)
print(s1.age)
print(s1.city)




#   __init__ Function

# Constructor 
# __init__ function is a special function that is used to initialize objects.


class Students:
    def __init__(self, name, age, city):
        self.name=name
        self.age=age 
        self.city=city

s1=Students("Aditya", 22, "Bangalore")
s2=Students("Sam Altman", 40, "San Francisco")
s3=Students("Elon Musk", 50, "New York")

print(s1.name, s1.age, s1.city)
print(s2.name, s2.age, s2.city)
print(s3.name, s3.age, s3.city)



