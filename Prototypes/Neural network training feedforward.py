import numpy as np
import csv
from timeit import default_timer as timer
weights01=[]; weights12=[]; weights23=[]; weights34=[]; bias1=[]; bias2=[]; bias3=[]; bias4=[]
def load():
    global weights01; global weights12; global weights23; global weights34
    global bias1; global bias2; global bias3; global bias4
    with open("Object-Identifer-ML\Weights\weights01.csv","r") as f:
        rows=csv.reader(f)
        for row in rows:
            if len(row) !=0:
                weights01.append(row)
    with open("Object-Identifer-ML\Weights\weights12.csv","r") as f:
        rows=csv.reader(f)
        for row in rows:
            if len(row) !=0:
                weights12.append(row)
    with open("Object-Identifer-ML\Weights\weights23.csv","r") as f:
        rows=csv.reader(f)
        for row in rows:
            if len(row) !=0:
                weights23.append(row)
    with open("Object-Identifer-ML\Weights\weights34.csv","r") as f:
        rows=csv.reader(f)
        for row in rows:
            if len(row) !=0:
                weights34.append(row)
    with open("Object-Identifer-ML/Weights/bias1.csv","r") as f:
        rows=csv.reader(f)
        for row in rows:
            if len(row) !=0:
                bias1.append(row)
    with open("Object-Identifer-ML/Weights/bias2.csv","r") as f:
        rows=csv.reader(f)
        for row in rows:
            if len(row) !=0:
                bias2.append(row)
    with open("Object-Identifer-ML/Weights/bias3.csv","r") as f:
        rows=csv.reader(f)
        for row in rows:
            if len(row) !=0:
                bias3.append(row)
    with open("Object-Identifer-ML/Weights/bias4.csv","r") as f:
        rows=csv.reader(f)
        for row in rows:
            if len(row) !=0:
                bias4.append(row)
def store():    
    global weights01; global weights12; global weights23; global weights34
    global bias1; global bias2; global bias3; global bias4
    with open("Object-Identifer-ML\Weights\weights01.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(weights01)
    with open("Object-Identifer-ML\Weights\weights12.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(weights12)
    with open("Object-Identifer-ML\Weights\weights23.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(weights23)
    with open("Object-Identifer-ML\Weights\weights34.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(weights34)
    with open("Object-Identifer-ML/Weights/bias1.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(bias1)
    with open("Object-Identifer-ML/Weights/bias2.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(bias2)
    with open("Object-Identifer-ML/Weights/bias3.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(bias3)
    with open("Object-Identifer-ML/Weights/bias4.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(bias4)
class MLfuntions:
    def __init__(self,index):
        objectnames=["Bike","Bottle","Cat","Chair","Flower"]
        objects=[]
        for i in objectnames: 
            for two in range(2):
                pixelarray=[]
                with open(f"Object-Identifer-ML/DataCSV/{i}/{i} ({index*2-two}).csv","r") as f:
                    csvthingy=csv.reader(f)
                    for row in csvthingy:
                        if len(row) !=0:
                            pixelarray.append(row)
                    objects.append(pixelarray)
        self.object=objects



        self.ObjectPerfectOuput=[]
        bike=[1,0,0,0,0];bottle=[0,1,0,0,0];cat=[0,0,1,0,0];chair=[0,0,0,1,0];flower=[0,0,0,0,1]
        self.ObjectPerfectOuput.append(bike);self.ObjectPerfectOuput.append(bike)
        self.ObjectPerfectOuput.append(bottle);self.ObjectPerfectOuput.append(bottle)
        self.ObjectPerfectOuput.append(cat);self.ObjectPerfectOuput.append(cat)
        self.ObjectPerfectOuput.append(chair);self.ObjectPerfectOuput.append(chair)
        self.ObjectPerfectOuput.append(flower);self.ObjectPerfectOuput.append(flower)
    def ReLU(self,x):
        return np.maximum(0,x)
    def ReLU_diff(self,x):
        return np.where(x<0,0,1)
    def sigmoid(self,x):
        return 1 / (1+np.exp(-x))
    def sigmoid_diff(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    def cost(self,position,output):
        costsum=0
        for i in range(5):
            costsum+=(position[i]-output[i])**2
        return costsum 
    def feedforward(self,inputnode,weight,bias):
        weightsmatrix=np.array(weight)
        inputvector=np.array(inputnode)
        bias=np.array(bias[0])
        weightsmatrix=weightsmatrix.T #Transposes weight matrix to allow for node calculation using matrix multiplication
        inputvector=inputvector.astype(float)
        weightsmatrix=weightsmatrix.astype(float)
        bias=bias.astype(float)
        inputvector=np.divide(inputvector,255)
        multipliedvalues=np.matmul(weightsmatrix,inputvector)
        outputnode=np.add(multipliedvalues,bias)
        correctedoutput=self.sigmoid(outputnode)
        return correctedoutput
    def feedforwardall(self,pos,weights01,bias1,weights12,bias2,weights23,bias3,weights34,bias4):
        node1=self.feedforward(self.object[pos][0],weights01,bias1)
        node2=self.feedforward(node1,weights12,bias2)
        node3=self.feedforward(node2,weights23,bias3)
        outputnode=self.feedforward(node3,weights34,bias4)
        cost=self.cost(self.ObjectPerfectOuput[pos],outputnode)
        return node1,node2,node3,outputnode,cost
    def feedforwardcostavg(self,weights01,bias1,weights12,bias2,weights23,bias3,weights34,bias4):
        costlist=[] #Initialises costlist
        for pos in range(10): #For all examples in epoch
            costlist.append(self.feedforwardall(pos,weights01,bias1,weights12,bias2,weights23,bias3,weights34,bias4)[4])
            #Applies feedforward function 10 times and adds to costlist
        return sum(costlist)/len(costlist) #Averages cost by summing and diving by number of examples

start=timer() #Starts timer 
load() #Loads weights and biases into program
MLfuncitonsinit=MLfuntions(1) #Initialises class for first 2 images in each list
costavg=MLfuncitonsinit.feedforwardcostavg(weights01,bias1,weights12,bias2,weights23,bias3,weights34,bias4)
#Applies feedforward function for first epoch
print(costavg) #Prints the cost average 
end=timer() #Ends timer
print(end-start," Seconds") #Finds difference in time

"""
MLfuncitonsinit=MLfuntions(1) 
node1,node2,node3,outputnode,cost=MLfuncitonsinit.feedforwardall(1,weights01,bias1,weights12,bias2,weights23,bias3,weights34,bias4)
print(outputnode)
print(cost)
start=timer()
_______________________________________________________________________
load()
MLfuncitonsinit=MLfuntions(1) 
costavg=MLfuncitonsinit.feedforwardcostavg(weights01,bias1,weights12,bias2,weights23,bias3,weights34,bias4)
print(costavg)
end=timer()
print(end-start," Seconds")
---------------------------------------------------------------------------------------------------------------
load()
MLfuncitonsinit=MLfuntions(1) 
node1,node2,node3,outputnode,cost=MLfuncitonsinit.feedforwardall(1,weights01,bias1,weights12,bias2,weights23,bias3,weights34,bias4)
print(outputnode)
print(cost)

----------------------------------------------------------------------------------------------------------------
node1=MLfuncitonsinit.feedforward(MLfuncitonsinit.object[2][0],weights01,bias1)
node2=MLfuncitonsinit.feedforward(node1,weights12,bias2)
node3=MLfuncitonsinit.feedforward(node2,weights23,bias3)
outputnode=MLfuncitonsinit.feedforward(node3,weights34,bias4)
cost=MLfuncitonsinit.cost(MLfuncitonsinit.ObjectPerfectOuput[2],outputnode)
print(outputnode)
print(cost)
"""