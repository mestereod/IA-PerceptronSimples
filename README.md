# IA-SimplePerceptron
## Use Example
```
#Training set 
training_set = [((1,1,1,1), 1), ((-1,1,-1,-1), 1), ((1,1,1,-1), -1), ((1,-1,-1,1), -1)]

#New Perceptron
ps = PS(training_set)

#Train Perceptron with training set
n_epoch = 10
ps.train(n_epoch)

#Print predict
for x, y in training_set:
    print("Input: {} Predict: {} Expected: {}".format(x, ps.predict(x), y))
   
#Print Weights
print("Weights:")
print(ps.getW())
```
