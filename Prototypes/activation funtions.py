import numpy as np
import csv
weights01=[]; weights12=[]; weights23=[]; weights34=[]; bias1=[]; bias2=[]; bias3=[]; bias4=[]
def load():
    global weights01; global weights12; global weights23; global weights34
    global bias1; global bias2; global bias3; global bias4
    with open("Object-Identifier-ML\Weights\weights01.csv","r") as f:
        rows=csv.reader(f)
        for row in rows:
            if len(row) !=0:
                weights01.append(row)
    with open("Object-Identifier-ML\Weights\weights12.csv","r") as f:
        rows=csv.reader(f)
        for row in rows:
            if len(row) !=0:
                weights12.append(row)
    with open("Object-Identifier-ML\Weights\weights23.csv","r") as f:
        rows=csv.reader(f)
        for row in rows:
            if len(row) !=0:
                weights23.append(row)
    with open("Object-Identifier-ML\Weights\weights34.csv","r") as f:
        rows=csv.reader(f)
        for row in rows:
            if len(row) !=0:
                weights34.append(row)
    with open("Object-Identifier-ML/Weights/bias1.csv","r") as f:
        rows=csv.reader(f)
        for row in rows:
            if len(row) !=0:
                bias1.append(row)
    with open("Object-Identifier-ML/Weights/bias2.csv","r") as f:
        rows=csv.reader(f)
        for row in rows:
            if len(row) !=0:
                bias2.append(row)
    with open("Object-Identifier-ML/Weights/bias3.csv","r") as f:
        rows=csv.reader(f)
        for row in rows:
            if len(row) !=0:
                bias3.append(row)
    with open("Object-Identifier-ML/Weights/bias4.csv","r") as f:
        rows=csv.reader(f)
        for row in rows:
            if len(row) !=0:
                bias4.append(row)
def store():    
    global weights01; global weights12; global weights23; global weights34
    global bias1; global bias2; global bias3; global bias4
    with open("Object-Identifier-ML\Weights\weights01.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(weights01)
    with open("Object-Identifier-ML\Weights\weights12.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(weights12)
    with open("Object-Identifier-ML\Weights\weights23.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(weights23)
    with open("Object-Identifier-ML\Weights\weights34.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(weights34)
    with open("Object-Identifier-ML/Weights/bias1.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(bias1)
    with open("Object-Identifier-ML/Weights/bias2.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(bias2)
    with open("Object-Identifier-ML/Weights/bias3.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(bias3)
    with open("Object-Identifier-ML/Weights/bias4.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(bias4)
class MLfuntions:
    def __init__(self,index):
        objectnames=["Bike","Bottle","Cat","Chair","Flower"]
        objects=[]
        for i in objectnames: 
            for two in range(2):
                pixelarray=[]
                with open(f"Object-Identifier-ML/DataCSV/{i}/{i} ({index*2-two}).csv","r") as f:
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

MLfuncitonsinit=MLfuntions(1)
#Graph
import matplotlib.pyplot as plt
x=np.linspace(-5,5,1000)
plt.figure("Activation functions",figsize=(20, 20))
plt.subplot(2, 1, 1)
plt.plot(x,MLfuncitonsinit.sigmoid(x),label="Sigmoid")
plt.plot(x,MLfuncitonsinit.sigmoid_diff(x),label="Sigmoid differential")
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(x,MLfuncitonsinit.ReLU(x),label="ReLU")
plt.plot(x,MLfuncitonsinit.ReLU_diff(x),label="ReLU differential")
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.show()

