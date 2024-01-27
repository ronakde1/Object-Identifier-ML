import numpy as np
import matplotlib.pyplot as plt
class MLfuntions:
    def __init__(self):
        self.inputvalues=np.array([0,1])
        self.ObjectPerfectOuput=np.array([1,0])
        self.learningrate=0.1
        self.weights01=np.random.normal(0, 1)
        self.bias=np.random.normal(0, 1)
        print(f"Original weights {self.weights01}")
        print(f"Original bias {self.bias}")
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
    def inv_sigmoid(self,x):
        return np.log(x / (1 - x))
    def backpropagation(self,output):
        error=self.ObjectPerfectOuput-output
        adjustments = error * self.sigmoid_diff(self.inv_sigmoid(output))
        self.weights01 += np.dot(self.inputvalues.T,adjustments) * self.learningrate
        self.bias += np.sum(adjustments) * self.learningrate
        
MLfuncitonsinit=MLfuntions()
costvalues=[]
sequentialnumbers=[]
for i in range(10000):
    outputnode,costs=MLfuncitonsinit.feedforwardall()
    MLfuncitonsinit.backpropagation(outputnode)
    costvalues.append(costs)
    sequentialnumbers.append(i+1)
print(f"New weight: {MLfuncitonsinit.weights01}");print(f"New bias: {MLfuncitonsinit.bias}")
print("Output for 0:",outputnode[0]);print("Output for 1:",outputnode[1])
print("Cost for final value",costs)
plt.figure("Cost function against epoch",figsize=(20, 20))
plt.subplot(2, 1, 1);plt.plot(sequentialnumbers,costvalues,label="Cost function against epoch");plt.legend()
plt.subplot(2, 1, 2);plt.plot(sequentialnumbers,np.log(costvalues),label="Logarithm of cost function against epoch");plt.legend()
plt.show()
