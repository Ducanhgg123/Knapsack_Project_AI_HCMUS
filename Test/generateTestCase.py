import random
import os
from path import Path
p="C:\\Users\\DUCANH\\Desktop\\AI\\Project1 - Knapsack\\TestCase"
Path(p).chdir()
print(os.getcwd())
f=open("input_7.txt","w")
W=50000
c=5
n=200
minWeight=20
maxWeight=1000
minValue=20
maxValue=1000
f.write(str(W)+'\n')
f.write(str(c)+'\n')
for i in range(n):
    f.write(str(random.randint(minWeight,maxWeight)))
    if i<n-1:
        f.write(',')
f.write('\n')
for i in range(n):
    f.write(str(random.randint(minValue,maxValue)))
    if i<n-1:
        f.write(',')

f.write('\n')
for i in range(n):
    f.write(str(random.randint(1,c)))
    if i<n-1:
        f.write(',')