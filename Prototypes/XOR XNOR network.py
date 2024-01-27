import numpy as np
import matplotlib.pyplot as plt
class MLfuntions:
    def __init__(self):
        self.inputvalues=np.array([[1,1],[0,0],[1,0],[0,1]])
        self.ObjectPerfectOuput=np.array([[1,0],[1,0],[0,1],[0,1]])
        self.learningrate=0.1
        self.weights01=np.random.normal(0, 1,size=(2,2))
        self.weights12=np.random.normal(0,1,size=(2,2))
        self.bias1=np.random.normal(0, 1,size=(2))
        self.bias2=np.random.normal(0, 1,size=(2))
        print(f"Original weights for first layer:\n{self.weights01}");print(f"Original biases for first layer:\n{self.bias1}")
        print(f"Original weights for second layer:\n{self.weights12}");print(f"Original biases for second layer:\n{self.bias2}")
    def sigmoid(self,x):
        return 1 / (1+np.exp(-x))
    def sigmoid_diff(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    def cost(self,perfectoutput,output):
        costsum=0
        for options in range(4):
            for outputnodes in range(2):
                costsum+=(perfectoutput[options][outputnodes]-output[options][outputnodes])**2
        return costsum/4 
    def feedforward(self,inputnode,weights,bias):
        multipliedvalues=np.matmul(weights,inputnode)
        outputnode=np.add(multipliedvalues,bias)
        correctedoutput=self.sigmoid(outputnode)
        return correctedoutput
    def feedforwardall(self):
        outputs=[]
        for i in self.inputvalues:
            node1=self.feedforward(i,self.weights01,self.bias1)
            outputnode=self.feedforward(node1,self.weights12,self.bias2)
            outputs.append(outputnode)
        costs=self.cost(outputs,self.ObjectPerfectOuput)
        return np.array(outputs),costs
MLfuncitonsinit=MLfuntions()
outputnode,costs=MLfuncitonsinit.feedforwardall()
print(f"\n Output node activations:\n{outputnode}\nCost: {costs}")

