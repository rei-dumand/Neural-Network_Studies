fruits = ["apple", "banana", "orange"]

def function(fruits):
    for x in fruits:
        print(x)

function(fruits)

class Animal:
    def __init__(self, specie, age):
        self.specie = specie
        self.age = age
    isSleeping = False
    isEating = True

Potato = Animal("dog", 5)

print(Potato.isSleeping)

