# IA-SimplePerceptron
## Use Example
```
#Training set 
training_set = [((1,1,1,1), 1), ((-1,1,-1,-1), 1), ((1,1,1,-1), -1), ((1,-1,-1,1), -1)]

#New Perceptron
n_epoch = 10
ps = PS(training_set, n_epoch)

#Train Perceptron with training set
ps.train()

#Print predict
for x, y in training_set:
    print("Input: {} Predict: {} Expected: {}".format(x, ps.predict(x), y))
   
#Print Weights
print("Weights:")
print(ps.getW())
```
