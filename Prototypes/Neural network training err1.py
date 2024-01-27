import numpy as np
import csv
weights01=[]; weights12=[]; weights23=[]; weights34=[]; bias1=[]; bias2=[]; bias3=[]; bias4=[]
def load():
    global weights01; global weights12; global weights23; global weights34
    global bias1; global bias2; global bias3; global bias4
    with open("Object-Identifer-ML\Weights\weights01.csv","r") as f:
        rows=csv.reader(f)
        for row in rows:
            if len(row) !=0:
                weights01.append(row)
        weights01=np.array(weights01)

    with open("Object-Identifer-ML\Weights\weights12.csv","r") as f:
        rows=csv.reader(f)
        for row in rows:
            if len(row) !=0:
                weights12.append(row)
        weights12=np.array(weights12)
    with open("Object-Identifer-ML\Weights\weights23.csv","r") as f:
        rows=csv.reader(f)
        for row in rows:
            if len(row) !=0:
                weights23.append(row)
        weights23=np.array(weights23)
    with open("Object-Identifer-ML\Weights\weights34.csv","r") as f:
        rows=csv.reader(f)
        for row in rows:
            if len(row) !=0:
                weights34.append(row)
        weights34=np.array(weights34)
    with open("Object-Identifer-ML/Weights/bias1.csv","r") as f:
        rows=csv.reader(f)
        for row in rows:
            if len(row) !=0:
                bias1.append(row)
        bias1=np.array(bias1[0])
    with open("Object-Identifer-ML/Weights/bias2.csv","r") as f:
        rows=csv.reader(f)
        for row in rows:
            if len(row) !=0:
                bias2.append(row)
        bias2=np.array(bias2[0])
    with open("Object-Identifer-ML/Weights/bias3.csv","r") as f:
        rows=csv.reader(f)
        for row in rows:
            if len(row) !=0:
                bias3.append(row)
        bias3=np.array(bias3[0])
    with open("Object-Identifer-ML/Weights/bias4.csv","r") as f:
        rows=csv.reader(f)
        for row in rows:
            if len(row) !=0:
                bias4.append(row)
        bias4=np.array(bias4[0])
def store():    
    global weights01; global weights12; global weights23; global weights34
    global bias1; global bias2; global bias3; global bias4
    with open("Object-Identifer-ML\Weights\weights01.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(weights01.tolist())
    with open("Object-Identifer-ML\Weights\weights12.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(weights12.tolist())
    with open("Object-Identifer-ML\Weights\weights23.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(weights23.tolist())
    with open("Object-Identifer-ML\Weights\weights34.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(weights34.tolist())
    with open("Object-Identifer-ML/Weights/bias1.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(bias1.tolist())
    with open("Object-Identifer-ML/Weights/bias2.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(bias2.tolist())
    with open("Object-Identifer-ML/Weights/bias3.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(bias3.tolist())
    with open("Object-Identifer-ML/Weights/bias4.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(bias4.tolist())
def store_fail(number):  
    global weights01; global weights12; global weights23; global weights34
    global bias1; global bias2; global bias3; global bias4
    with open(f"Object-Identifer-ML\Weights\Failweights\{number}weights01.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(weights01.tolist())
    with open(f"Object-Identifer-ML\Weights\Failweights\{number}weights12.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(weights12.tolist())
    with open(f"Object-Identifer-ML\Weights\Failweights\{number}weights23.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(weights23.tolist())
    with open(f"Object-Identifer-ML\Weights\Failweights\{number}weights34.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(weights34.tolist())
    with open(f"Object-Identifer-ML/Weights/Failweights/{number}bias1.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(bias1.tolist())
    with open(f"Object-Identifer-ML/Weights/Failweights/{number}bias2.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(bias2.tolist())
    with open(f"Object-Identifer-ML/Weights/Failweights/{number}bias3.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(bias3.tolist())
    with open(f"Object-Identifer-ML/Weights/Failweights/{number}bias4.csv","w") as f:
        csv_writer=csv.writer(f)
        csv_writer.writerows(bias4.tolist())
def refreshvariables():
    with open("Object-Identifer-ML/Weights/weights01.csv","w") as f:
        csv_writer=csv.writer(f)

        weights01=np.random.normal(0, 1, size=(40000,20)).tolist()
        csv_writer.writerows(weights01)

    with open('Object-Identifer-ML/Weights/bias1.csv',"w") as f:
        csv_writer=csv.writer(f)
        bias1=np.random.normal(0, 1, size=(20)).tolist()
        csv_writer.writerow(bias1)

    with open("Object-Identifer-ML/Weights/weights12.csv","w") as f:
        csv_writer=csv.writer(f)
        weights12=np.random.normal(0, 1, size=(20,20)).tolist()
        csv_writer.writerows(weights12)

    with open('Object-Identifer-ML/Weights/bias2.csv',"w") as f:
        csv_writer=csv.writer(f)
        bias2=np.random.normal(0, 1, size=(20)).tolist()
        csv_writer.writerow(bias2)

    with open("Object-Identifer-ML/Weights/weights23.csv","w") as f:
        csv_writer=csv.writer(f)
        weights23=np.random.normal(0, 1, size=(20,20)).tolist()
        csv_writer.writerows(weights23)

    with open('Object-Identifer-ML/Weights/bias3.csv',"w") as f:
        csv_writer=csv.writer(f)
        bias3=np.random.normal(0, 1, size=(20)).tolist()
        csv_writer.writerow(bias3)

    with open("Object-Identifer-ML/Weights/weights34.csv","w") as f:
        csv_writer=csv.writer(f)
        weights34=np.random.normal(0, 1, size=(20, 5)).tolist()
        csv_writer.writerows(weights34)

    with open('Object-Identifer-ML/Weights/bias4.csv',"w") as f:
        csv_writer=csv.writer(f)
        bias4=np.random.normal(0, 1, size=(5)).tolist()
        csv_writer.writerow(bias4)
class MLfuntions:
    def __init__(self,index,w01,w12,w23,w34,b1,b2,b3,b4):
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
        self.learningrate=0.5
        self.weights01=w01;self.weights12=w12;self.weights23=w23;self.weights34=w34
        self.bias1=b1;self.bias2=b2;self.bias3=b3;self.bias4=b4
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
    def listall(self):
        return self.weights01,self.weights12,self.weights23,self.weights34,self.bias1,self.bias2,self.bias3,self.bias4
    def sigmoid(self,x):
        return 1 / (1+np.exp(-x))
    def sigmoid_diff(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    def inv_sigmoid(self,x):
        return np.log(x / (1 - x))
    def cost(self,position,output):
        costsum=[]
        for i in range(5):
            costsum.append((position[i]-output[i])**2)
        return np.array(costsum)
    def totalcost(self,perfectoutput,outputnode):
        costsumsum=np.array([0.0,0.0,0.0,0.0,0.0])
        for i in range(10):
            costsumsum+=self.cost(perfectoutput[i],outputnode[i])
        return costsumsum
    def feedforward(self,inputnode,weightsmatrix,bias):
        inputvector=np.array(inputnode)
        weightsmatrix=weightsmatrix.T #Transposes weight matrix to allow for node calculation using matrix multiplication
        inputvector=inputvector.astype(float)
        weightsmatrix=weightsmatrix.astype(float)
        bias=bias.astype(float)
        inputvector=np.divide(inputvector,255)
        multipliedvalues=np.matmul(weightsmatrix,inputvector)
        outputnode=np.add(multipliedvalues,bias)
        correctedoutput=self.sigmoid(outputnode)
        return correctedoutput
    def feedforwardall(self):
        node1=[];node2=[];node3=[];outputnode=[]
        for pos in range(10):

            node1.append(self.feedforward(self.object[pos][0],self.weights01,self.bias1))
            node2.append(self.feedforward(node1[pos],self.weights12,self.bias2))
            node3.append(self.feedforward(node2[pos],self.weights23,self.bias3))
            outputnode.append(self.feedforward(node3[pos],self.weights34,self.bias4))
        totalcost=self.totalcost(self.ObjectPerfectOuput,outputnode)
        return node1,node2,node3,outputnode,totalcost
    def backpropagation(self,node1,node2,node3,outputnode):#Applies the backpropagation function
        #Sets adjustments as 0
        adjustmentweights01=0;adjustmentweights12=0;adjustmentweights23=0;adjustmentweights34=0;adjustmentsbias1=0;adjustmentsbias2=0;adjustmentsbias3=0;adjustmentsbias4=0
        numexamples=len(outputnode) #Number of input sets 
        for i in range(numexamples): #For each input set

            error4=self.ObjectPerfectOuput[i]-outputnode[i] #Calculates between error between output and expected output for a specific example
            adjustmentweights34+=self.learningrate*np.outer(error4*self.sigmoid_diff(self.inv_sigmoid(outputnode[i])),node3[i])
            #Inverses sigmoid function -> applies differential -> applies hadamard product with error -> multiplies hidden layer by result -> multiplies by learning rate
            adjustmentsbias4+=self.learningrate*error4*self.sigmoid_diff(self.inv_sigmoid(outputnode[i]))
            #Inverses sigmoid function -> applies differential -> applies hadamard product with error -> multiplies by learning rate
            
            error3=np.matmul(self.weights34.T,error4) #Propagating the error backwards using tranpose of weight matrix 
            adjustmentweights23+=self.learningrate*np.outer(error3*self.sigmoid_diff(self.inv_sigmoid(node3[i])),node2[i])
            #Applies weight adjustment algorithm to hidden layer (line 225)
            adjustmentsbias3+=self.learningrate*error3*self.sigmoid_diff(self.inv_sigmoid(node3[i]))
            #Applies bias adjustment algorithm to hidden layer (line 227)

            #Repeats previous steps for between 2nd layer and 3rd layer
            error2=np.matmul(self.weights23.T,error3)
            adjustmentweights12+=self.learningrate*np.outer(error2*self.sigmoid_diff(self.inv_sigmoid(node2[i])),node1[i])
            adjustmentsbias2+=self.learningrate*error2*self.sigmoid_diff(self.inv_sigmoid(node2[i]))

            #Repeats previous steps for between 1st layer and 2nd layer
            error1=np.matmul(self.weights12.T,error2)
            adjustmentweights01+=self.learningrate*np.outer(error1*self.sigmoid_diff(self.inv_sigmoid(node1[i])),self.object[i][0])
            adjustmentsbias1+=self.learningrate*error1*self.sigmoid_diff(self.inv_sigmoid(node1[i]))

        #To average the adjustments from each example (each example gives a normalised adjustment so no single example increases cost for entire example)
        self.weights34+=adjustmentweights34/numexamples
        self.weights23+=adjustmentweights23/numexamples
        self.weights12+=adjustmentweights12/numexamples
        self.weights01+=adjustmentweights01/numexamples
        self.bias1+=adjustmentsbias1/numexamples
        self.bias2+=adjustmentsbias2/numexamples
        self.bias3+=adjustmentsbias3/numexamples
        self.bias4+=adjustmentsbias4/numexamples

#----------------------------------------------------MAIN----------------------------------------------------
epochrepeats=5
epochnumber=80
#Initialise variables
load()
for i in range(epochrepeats):
    for i in range(1,epochnumber):
        #Initialises class
        MLfuncitonsinit=MLfuntions(i,weights01,weights12,weights23,weights34,bias1,bias2,bias3,bias4)  
        #Training for specific epoch
        node1,node2,node3,outputnode,totalcost=MLfuncitonsinit.feedforwardall()
        MLfuncitonsinit.backpropagation(node1,node2,node3,outputnode)
        weights01,weights12,weights23,weights34,bias1,bias2,bias3,bias4=MLfuncitonsinit.listall()
    #Cost at point in time (progress)
    print(np.sum(totalcost))
    #New epoch cycle
    store()

