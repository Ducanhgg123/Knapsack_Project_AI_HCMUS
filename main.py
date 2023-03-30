import random
from typing import List
import heapq
class Item:
    def __init__(self, weight, value, c):
        self.weight = weight
        self.value = value
        self.c = c


class Individual:
    bits = []

    def __init__(self, bits: list[int]):
        self.bits = bits.copy()

    def fitness(self) -> float:
        totalWeight = 0
        totalValue = 0
        classCheck = set()
        for i in range(len(self.bits)):
            totalWeight = totalWeight + self.bits[i] * KNAPSACK[i].weight
            totalValue = totalValue + self.bits[i] * KNAPSACK[i].value
            classCheck.add(KNAPSACK[i].c)
        if totalWeight > MAX_WEIGHT or len(classCheck) != NUM_CLASS:
            return 0
        return totalValue
class Knapsack:
    def __init__(self) -> None:
        self.capacity = self.input()["capacity"]
        self.num_of_classes = self.input()["num_of_classes"]
        self.weights = self.input()["weights"]
        self.values = self.input()["values"]
        self.labels = self.input()["labels"]

    def input(self):
        capacity = 0
        num_of_classes = 0
        weights = []
        values = []
        labels = []

        f = open("input.txt", "r")
        lines = f.readlines()
        capacity = int(lines[0])
        num_of_classes = int(lines[1])
        weights = [int(i) for i in lines[2].split(',')]
        values = [int(i) for i in lines[3].split(',')]
        labels = [int(i) for i in lines[4].split(',')]
        # Sort the input
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                if values[i] / weights[i] < values[j] / weights[j]:
                    temp = values[i]
                    values[i] = values[j]
                    values[j] = temp

                    temp = weights[i]
                    weights[i] = weights[j]
                    weights[j] = temp

                    temp = labels[i]
                    labels[i] = labels[j]
                    labels[j] = temp
        return {
            "capacity": capacity,
            "num_of_classes": num_of_classes,
            "weights": weights,
            "values": values,
            "labels": labels
        }

    def brute(self):
        n = len(self.values)
        length_unique_labels = len(set(self.labels))
        res = 0
        curLabels = []
        arrBits = [0] * n
        resBits = [0] * n

        def helper(capacity, curValue, i):
            nonlocal res, arrBits, resBits
            if i == n or capacity == 0:
                if curValue > res and len(set(curLabels)) == length_unique_labels:
                    res = curValue
                    resBits = arrBits.copy()
                return

            # include the i-th item
            if self.weights[i] <= capacity:
                curLabels.append(self.labels[i])
                arrBits[i] = 1
                helper(capacity - self.weights[i], curValue + self.values[i], i + 1)
                arrBits[i] = 0
                curLabels.remove(self.labels[i])

            # exclude the i-th item
            helper(capacity, curValue, i + 1)

        helper(self.capacity, 0, 0)
        return [res, resBits]

    def getLabelsLength(self, input):
        labels = set()
        for i in range(len(input)):
            if input[i] == 1:
                labels.add(self.labels[i])
        return len(labels)

    def branch_and_bound(self):
        n = len(self.values)
        length_unique_labels = len(set(self.labels))
        res = 0
        curLabels = []
        arrBits = [0] * n
        resBits = [0] * n

        def calculateBranch(capacity: float, pos: int) -> int:
            listItem = []
            weight = self.weights.copy()
            value = self.values.copy()
            labels = self.labels.copy()
            for i in range(len(weight)):
                listItem.append((weight[i], value[i], labels[i], value[i] / weight[i]))

            # Calculate the branch
            Sum = 0
            i = 0
            while capacity > 0 and i < n:
                if capacity > weight[i]:
                    Sum = Sum + listItem[i][3] * listItem[i][0]
                    capacity = capacity - listItem[i][0]
                else:
                    Sum = Sum + listItem[i][3] * capacity
                    break
                i = i + 1
            return Sum

        def helper(capacity, curValue, i):
            nonlocal res, arrBits, resBits
            if i == n or capacity == 0:
                if curValue > res and len(set(curLabels)) == length_unique_labels:
                    res = curValue
                    resBits = arrBits.copy()
                return

            if curValue + calculateBranch(capacity, i) < res:
                return
            if self.weights[i] <= capacity:
                curLabels.append(self.labels[i])
                arrBits[i] = 1
                helper(capacity - self.weights[i], curValue + self.values[i], i + 1)
                arrBits[i] = 0
                curLabels.remove(self.labels[i])

            # exclude the i-th item
            helper(capacity, curValue, i + 1)

        helper(self.capacity, 0, 0)
        return [res, resBits]

W = 5  # number of nodes taken after a turn of generation
runs = 200
exploredSet = dict()
maxVal = 0
resState = []
recursiveCount = 0
recursiveCalls = 10
class BeamSearch:
    listState: []
    def __init__(self):
        self.listState = []

    def genInitState(self, problem: Knapsack):
        for n in range(W):
            state = []
            for i in range(len(problem.values)):
                value = random.randint(0, 1)
                state.append(value)
            self.listState.append(state)

    def fitness(self, problem: Knapsack, state: []):
        totalVal = 0
        totalWeight = 0
        classCheck = set()
        for i in range(len(state)):
            if state[i] == 1:
                totalVal += problem.values[i]
                totalWeight += problem.weights[i]
                classCheck.add(problem.labels[i])

        if totalWeight > problem.capacity or len(classCheck) != problem.num_of_classes:
            return 0
        return -totalVal
    def genSuccessors(self, problem: Knapsack, state: []):
        successorsList = []
        for i in range(len(state)):
            successor = state.copy()
            successor[i] = 1 - successor[i]
            successorsList.append(successor)
        return successorsList

    def run(self, problem: Knapsack):
        self.genInitState(problem)
        q = []
        heapq.heapify(q)
        n = 0
        global maxVal
        global resState
        global exploredSet
        global recursiveCount
        global recursiveCalls

        if recursiveCount == recursiveCalls:
            return maxVal, resState
        else:
            recursiveCount += 1

        for state in self.listState:
            heapq.heappush(q, (self.fitness(problem, state), state))
            exploredSet[str(state)] = True
            if maxVal < -self.fitness(problem, state):
                maxVal = -self.fitness(problem, state)
                resState = state.copy()

        while n < runs and not len(q) == 0:
            initStates = []
            for i in range(W):
                if len(q) == 0:
                    self.run(problem)
                else:
                    fit,state = heapq.heappop(q)
                initStates.append((fit,state))

            q=[]

            for item in initStates:
                fit,state = item
                successors = self.genSuccessors(problem, state)
                for sc in successors:
                    f = self.fitness(problem, sc)
                    if str(sc) not in exploredSet and f <= fit:
                        if maxVal < -f:
                            maxVal = -f
                            resState = sc.copy()
                        exploredSet[str(sc)]=True
                        heapq.heappush(q,(f, sc))

            n += 1
        return maxVal, resState

MAX_GEN=300
INIT_POPULATION=500
CROSSOVER_RATE = 0.53
MUTATION_RATE = 0.013
REPRODUCTION_RATE = 0.15
class GeneticAlgorithm:
    population=[]
    num_of_classes=0
    capacity=0
    weights=[]
    values=[]
    labels=[]
    maxVal=0
    maxRes=[]
    def __init__(self):
        ks=Knapsack()
        source=ks.input()
        self.weights=source["weights"]
        self.values=source["values"]
        self.labels=source["labels"]
        self.num_of_class=source["num_of_classes"]
        self.capacity=source["capacity"]

    def generateInitalPopulation(self):
        count=0
        while count<INIT_POPULATION:
            count=count+1
            bits=[
                random.choice([0,1])
                for _ in self.weights
            ]
            self.population.append(bits)

    def calculateFitness(self,bits:list[int]):
        totalWeight=0
        totalValue=0
        classCheck=set()
        for i in range(len(self.weights)):
            totalWeight=totalWeight+bits[i]*self.weights[i]
            totalValue=totalValue+bits[i]*self.values[i]
            classCheck.add(self.labels[i])

        if len(classCheck)!=self.num_of_class or totalWeight>self.capacity:
            return 0
        return totalValue

    def selection(self):
        random.shuffle(self.population)
        parents=[]
        i=0
        while i<len(self.population):
            if i+1==len(self.population):
                break
            if self.calculateFitness(self.population[i])>self.calculateFitness(self.population[i+1]):
                parents.append(self.population[i])
            else:
                parents.append(self.population[i+1])
            i=i+2
        self.population=parents.copy()

    def crossover(self,father: list[int],mother: list[int]):
        n=len(father)
        child1=father[:n//2]+mother[n//2:]
        child2=mother[:n//2]+father[n//2:]
        return child1,child2

    def crossoverMethod(self):
        children=[]
        i=0
        while i<len(self.population):
            if i+1==len(self.population):
                break
            child1,child2=self.crossover(self.population[i],self.population[i+1])
            if self.calculateFitness(child1)>self.maxVal:
                self.maxVal=self.calculateFitness(child1)
                self.maxRes=child1

            if self.calculateFitness(child2)>self.maxVal:
                self.maxVal=self.calculateFitness(child2)
                self.maxRes=child2

            children.append(child1)
            children.append(child2)
            i=i+2
        for child in children:
            self.population.append(child)

    def mutate(self,individual:list[int]) -> list[int]:
        pos=random.randint(0,len(individual)-1)
        individual[pos]=1-individual[pos]
        return individual

    def mutatationMethod(self):
        for i in range(len(self.population)):
            if random.random()<MUTATION_RATE:
                self.population[i]=self.mutate(self.population[i])

    def run(self):
        self.generateInitalPopulation()
        for i in range(MAX_GEN):
            self.selection()
            self.crossoverMethod()
            self.mutatationMethod()

        vMax=0
        for individual in self.population:
            vMax=max(vMax,self.calculateFitness(individual))
        return self.maxVal,self.maxRes

ks=Knapsack()
print(ks.brute())
print(ks.branch_and_bound())
Gen=GeneticAlgorithm()
print(Gen.run())

beamSearch=BeamSearch()
print(beamSearch.run(ks))











