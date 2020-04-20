import numpy as np
import matplotlib.pyplot as plt 

class PS:
    def __init__(self, training_set):
        self.training_set = training_set
        self.b = 0
        self.eta = 1.0
        self.epoch = 10
        self.errors = []
        self.w = np.zeros(len(training_set[0][0]))

    def step_function(self,x):
        return -1 if x < 0 else 1


    def predict(self,input):
        summ = sum(input * self.w) + self.b
        return self.step_function(summ)


    def train(self):
        for _ in range(self.epoch):
            for x, y in self.training_set:
                summ = sum(x * self.w) + self.b
                predict = self.step_function(summ)
                error = y - predict
                #print("X: {} Y: {} Pred: {} Error: {}".format(x, y, predict, error))

                self.errors.append(error)
                for index, value in enumerate(x):
                    #print("Index: {} Value: {}".format(index, value))
                    self.w[index] += self.eta * error * value
                    self.b += self.eta * error
    
    def getW(self):
        return self.w

training_set = [((1,1,1,1), 1), ((-1,1,-1,-1), 1), ((1,1,1,-1), -1), ((1,-1,-1,1), -1)]
ps = PS(training_set)
ps.train()

for x, y in training_set:
    print("Input: {} Predict: {} Expected: {}".format(x, ps.predict(x), y))
print("Weights:")
print(ps.getW())