#Doesn't do as expected
import numpy as np
import matplotlib.pyplot as plt
class MLfuntions:
    def __init__(self,learningrate):
        self.inputvalues=np.array([0,1])
        self.ObjectPerfectOuput=np.array([1,0])
        self.learningrate=learningrate
        self.weights01=0.5
        self.bias=0.5
    def sigmoid(self,x):
        return 1 / (1+np.exp(-x))
    def sigmoid_diff(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    def cost(self,perfectoutput,output):
        costsum=0
        for i in range(2):
            costsum+=(perfectoutput[i]-output[i])**2
        return costsum 
    def feedforward(self,inputnode,weight,bias):
        outputnode=inputnode*weight+bias
        correctedoutput=self.sigmoid(outputnode)
        return correctedoutput
    def feedforwardall(self):
        outputs=[]
        for i in self.inputvalues:
            outputs.append(self.feedforward(i,self.weights01,self.bias))
        costs=self.cost(outputs,self.ObjectPerfectOuput)
        return np.array(outputs),costs
    def backpropagation(self,output):
        error=self.ObjectPerfectOuput-output
        adjustments = error * self.sigmoid_diff(output)
        self.weights01 += np.dot(self.inputvalues.T,adjustments) * self.learningrate
        self.bias += np.sum(adjustments) * self.learningrate

cost_threshold=0.001
learning_rates=[]
epoch_number=[]
for x in range(5000):
    x/=10

    MLfuncitonsinit=MLfuntions(x)
    for i in range(10000):
        outputnode,costs=MLfuncitonsinit.feedforwardall()
        MLfuncitonsinit.backpropagation(outputnode)
        if costs < cost_threshold:
            learning_rates.append(x)
            epoch_number.append(i+1)
            break



plt.figure("Cost function against epoch",figsize=(10, 10))
plt.subplot(2, 1, 1);plt.plot(learning_rates,epoch_number,label="Cost function against epoch");plt.legend()

plt.show()
