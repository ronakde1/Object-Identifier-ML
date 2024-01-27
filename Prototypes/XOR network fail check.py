import numpy as np
import matplotlib.pyplot as plt
class MLfuntions:
    def __init__(self):
        self.inputvalues=np.array([[1,1],[0,0],[1,0],[0,1]])
        self.ObjectPerfectOuput=np.array([[0,1],[0,1],[1,0],[1,0]])
        self.learningrate=0.5
        self.weights01=np.random.normal(0,1,size=(2,2))
        self.weights12=np.random.normal(0,1,size=(2,2))
        self.bias1=np.random.normal(0, 1,size=(2))
        self.bias2=np.random.normal(0, 1,size=(2))
        print(f"Original weights for first layer: {self.weights01}");print(f"Original biases for first layer: {self.bias1}")
        print(f"Original weights for second layer: {self.weights12}");print(f"Original biases for second layer: {self.bias2}\n")
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
    def inv_sigmoid(self,x):
        return np.log(x / (1 - x))
    def feedforward(self,inputnode,weights,bias):
        multipliedvalues=np.matmul(weights,inputnode)
        outputnode=np.add(multipliedvalues,bias)
        correctedoutput=self.sigmoid(outputnode)
        return correctedoutput
    def feedforwardall(self):
        node1s=[]
        outputs=[]
        for i in self.inputvalues:
            node1=self.feedforward(i,self.weights01,self.bias1)
            outputnode=self.feedforward(node1,self.weights12,self.bias2)
            outputs.append(outputnode)
            node1s.append(node1)
        costs=self.cost(outputs,self.ObjectPerfectOuput)
        return np.array(node1s),np.array(outputs),costs
    def backpropagation(self,nodes1,outputs):
        adjustmentweights01=0;adjustmentweights12=0;adjustmentsbias1=0;adjustmentsbias2=0
        numexamples=len(outputs)
        for i in range(numexamples):
            error2=self.ObjectPerfectOuput[i]-outputs[i]
            adjustmentweights12+=self.learningrate*np.outer(error2*self.sigmoid_diff(self.inv_sigmoid(outputs[i])),nodes1[i])
            adjustmentsbias2+=self.learningrate*error2*self.sigmoid_diff(self.inv_sigmoid(outputs[i]))

            error1=np.matmul(self.weights12.T,error2) #Propagating the error backwards
            adjustmentweights01+=self.learningrate*np.outer(error1*self.sigmoid_diff(self.inv_sigmoid(nodes1[i])),np.array(self.inputvalues[i]))
            adjustmentsbias1+=self.learningrate*error2*self.sigmoid_diff(self.inv_sigmoid(nodes1[i]))

        self.weights01+=adjustmentweights01/numexamples #To average the adjustments from each example
        self.weights12+=adjustmentweights12/numexamples
        self.bias1+=adjustmentsbias1/numexamples
        self.bias2+=adjustmentsbias2/numexamples
MLfuncitonsinit=MLfuntions() #Initialises the class
tcounter=0 #Allows for cost against epoch cycle to be shown (increases by 1 for every fail)
costvalues=[] #Creates a list to store cost values to compare against
sequentialnumbers=[] #Initilises a list of sequential numbers to plot against the cost
while True: #Will not stop unless program's cost is below 0.2 at the end of the traning process
    for i in range(10000): #Runs through 10000 epochs of training
        node1,outputnode,costs=MLfuncitonsinit.feedforwardall() #Applies the feedforward function to inputs, and outputs the cost
        MLfuncitonsinit.backpropagation(node1,outputnode) #Applies the backpropagation algoirthm to the previous output and hidden layer
        costvalues.append(costs) #Adds cost to list to plot against
        sequentialnumbers.append(i+1+tcounter*10000) #Produces a list of sequential numbers
    if costs>0.2: #Checks if cost is below 0.2
        print("fail",costs) 
        MLfuncitonsinit=MLfuntions() #Reinitalises class to rerandomise set of weights and biases
        tcounter+=1 #Adds 10,000 to sequential counter
        print() 
    else:
        break #If successful, breaks out of loop

print(f"New weights for first layer: {MLfuncitonsinit.weights01}");print(f"New biases for first layer: {MLfuncitonsinit.bias1}") #Prints weight after traning process
print(f"New weights for second layer: {MLfuncitonsinit.weights12}");print(f"New biases for second layer: {MLfuncitonsinit.bias2}") #Prints bias after training process
#Prints result for each input after training
print("Output for 11:",outputnode[0]);print("Output for 00:",outputnode[1])
print("Output for 10:",outputnode[2]);print("Output for 01:",outputnode[3])
print("Cost for final value",costs) #Represents how well the network has performed (lower is better)
plt.figure("Cost function against epoch",figsize=(10, 10))  #Initialises graph to plot
plt.plot(sequentialnumbers,costvalues,label="Cost function against epoch");plt.legend() #Plots cost against epoch number
plt.show()#Shows graph