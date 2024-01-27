from PIL import Image
import numpy as np
import csv
from datetime import datetime
import os
import pyttsx3
import time
def text_to_speech(text,objn,activate):
    engine = pyttsx3.init()
    if activate == "Y":
        rate = engine.getProperty('rate')
        ratechange=int(objn*10-50)
        engine.setProperty('rate', rate+ratechange)
    engine.say(text)
    engine.runAndWait()

def pauseyn(text):
    if text == "Y":
        time.sleep(1)

from picamera import PiCamera
camera = PiCamera()
camera.capture('Object-Identifer-ML\Capturelocation\IMG1.jpg')

with open("Object-Identifer-ML\Log\log.txt", 'r') as file: #Opens file in read mode
    #Validation check to ensure file contains number
    try:
        current_value = int(file.read().strip()) #Removes all whitespace and only reads the number
    except TypeError:
        current_value = 0
new_value = current_value + 1 #Increments the current value by 1
with open("Object-Identifer-ML\Log\log.txt", 'w') as file: #Opens file in write mode
    file.write(str(new_value)) #Writes value into file over the original value
now = datetime.now()

with open("Object-Identifer-ML\generalvariables\generalvariable1.txt", 'r') as file: #Opens file in read mode
    activate = str(file.read().strip()) #Removes all whitespace and only reads the value
    
    #Validation to ensure that activation is in correct format. If activation is not in correct form, then defaults to N and resets this.
    if activate == "Y": 
        pass
    elif activate == "N":
        pass
    else:
        activate= "N"
        print("Voice rate change reset to inactive")
        with open("Object-Identifer-ML\generalvariables\generalvariable1.txt", 'w') as file: #Opens file in write mode
            file.write(str(activate)) #Writes value into file over the original value

with open("Object-Identifer-ML\generalvariables\generalvariable2.txt", 'r') as file: #Opens file in read mode
    activatesleep = str(file.read().strip()) #Removes all whitespace and only reads the value
    
    #Validation to ensure that activation is in correct format. If activation is not in correct form, then defaults to N and resets this.
    if activatesleep == "Y": 
        pass
    elif activatesleep == "N":
        pass
    else:
        activatesleep = "N"
        print("Voice pause reset to inactive")
        with open("Object-Identifer-ML\generalvariables\generalvariable2.txt", 'w') as file: #Opens file in write mode
            file.write(str(activate)) #Writes value into file over the original value

camera.capture(f'Object-Identifer-ML\Log\IMG{current_value}{now}.jpg')

im=Image.open(f"Object-Identifer-ML\Capturelocation\IMG1.jpg") #Calls image location
im=im.resize((200,200)) #Resizes image
im=im.convert("L") #Turns image greyscale

pixels = im.load() #Loads image into pixels
all_pixels = [] 
for x in range(200):
    for y in range(200):
        cpixel = pixels[x, y] #Stores pixel values in structured data type
        all_pixels.append(cpixel)
with open(f"Object-Identifer-ML\DataCSV\IMG.csv","w") as f:
    csv_writer=csv.writer(f)
    csv_writer.writerow(all_pixels) #Writes pixel values


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
class MLfuntions:
    def __init__(self,w01,w12,w23,w34,b1,b2,b3,b4):
        pixelarray=[]
        with open("Object-Identifer-ML\DataCSV\IMG.csv","r") as f:
            csvthingy=csv.reader(f)
            for row in csvthingy:
                if len(row) !=0:
                    pixelarray.append(row)
        self.object=np.array(pixelarray,np.float64)
        self.weights01=w01.astype(float)
        self.weights12=w12.astype(float)
        self.weights23=w23.astype(float)
        self.weights34=w34.astype(float)
        self.bias1=b1.astype(float)
        self.bias2=b2.astype(float)
        self.bias3=b3.astype(float)
        self.bias4=b4.astype(float)
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

        inputvector=np.divide(inputvector,255)
        multipliedvalues=np.matmul(weightsmatrix,inputvector)
        outputnode=np.add(multipliedvalues,bias)
        correctedoutput=self.sigmoid(outputnode)
        return correctedoutput
    def feedforwardall(self):

        node1=self.feedforward(self.object,self.weights01,self.bias1)
        node2=self.feedforward(node1,self.weights12,self.bias2)
        node3=self.feedforward(node2,self.weights23,self.bias3)
        outputnode=self.feedforward(node3,self.weights34,self.bias4)
        return node1,node2,node3,np.array(outputnode)

#----------------------------------------------------MAIN----------------------------------------------------

load()
objectnames=["Bike","Bottle","Cat","Chair","Flower"]
MLfuncitonsinit=MLfuntions(weights01,weights12,weights23,weights34,bias1,bias2,bias3,bias4)  
node1,node2,node3,outputnode=MLfuncitonsinit.feedforwardall()
print()
os.remove("Object-Identifer-ML\Capturelocation\IMG1.jpg")
os.remove("Object-Identifer-ML\Capturelocation\IMG1.csv")


text_to_speech(objectnames[outputnode.argmax()],outputnode.argmax(),activate)
pauseyn(activatesleep)
