import numpy as np


class modular_functions():
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
    
    def cost(self,perfectoutput,output):
        costsum=0
        for i in range(2):
            costsum+=(perfectoutput[i]-output[i])**2
        return costsum 
    def costavg(self,x,numoutputs):
        return x/numoutputs

    def totallost(self,perfectoutput,outputnode):
        costsumsum=np.array([0.0,0.0,0.0,0.0,0.0])
        for i in range(10):
            costsumsum+=self.loss(perfectoutput[i],outputnode[i])
        return costsumsum
    

    