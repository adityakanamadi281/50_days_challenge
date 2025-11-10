class Student:
    college_name = "World's Reality University"

    def __init__(self,name,marks):
        self.name = name
        self.marks = marks
        print("Adding new students to database")

s1=Student("Aditya",35)
print(s1.name, s1.marks)

s2=Student("Elon",45)
s3=Student("Sam", 55)
print(s2.name, s2.marks)
print(s3.name, s3.marks)

class student:

    def __init__(self,name,marks):
        self.name = name
        self.marks = marks

    def avg(self):
        sum = 0
        for i in self.marks:
            sum += i
        print("Average marks of ",self.name,"is",sum/3)

s1=student("Aditya", [35,35,36])
s1.avg()
