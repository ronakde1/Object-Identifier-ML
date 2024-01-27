import csv
import numpy as np
#Loads libraries

with open("Object-Identifier-ML/Weights/weights01.csv","w") as f: #Creates CSV file
     csv_writer=csv.writer(f) #Turns into write mode
     weights01=np.random.normal(0, 1, size=(40000,20)).tolist() #Creates weights with normal distribution
     csv_writer.writerows(weights01) #Writes weights

with open('Object-Identifier-ML/Weights/bias1.csv',"w") as f:
     csv_writer=csv.writer(f)
     bias1=np.random.normal(0, 1, size=(20)).tolist()
     csv_writer.writerow(bias1)

with open("Object-Identifier-ML/Weights/weights12.csv","w") as f:
     csv_writer=csv.writer(f)
     weights12=np.random.normal(0, 1, size=(20,20)).tolist()
     csv_writer.writerows(weights12)

with open('Object-Identifier-ML/Weights/bias2.csv',"w") as f:
     csv_writer=csv.writer(f)
     bias2=np.random.normal(0, 1, size=(20)).tolist()
     csv_writer.writerow(bias2)

with open("Object-Identifier-ML/Weights/weights23.csv","w") as f:
     csv_writer=csv.writer(f)
     weights23=np.random.normal(0, 1, size=(20,20)).tolist()
     csv_writer.writerows(weights23)

with open('Object-Identifier-ML/Weights/bias3.csv',"w") as f:
     csv_writer=csv.writer(f)
     bias3=np.random.normal(0, 1, size=(20)).tolist()
     csv_writer.writerow(bias3)

with open("Object-Identifier-ML/Weights/weights34.csv","w") as f:
     csv_writer=csv.writer(f)
     weights34=np.random.normal(0, 1, size=(20, 5)).tolist()
     csv_writer.writerows(weights34)

with open('Object-Identifier-ML/Weights/bias4.csv',"w") as f:
     csv_writer=csv.writer(f)
     bias4=np.random.normal(0, 1, size=(5)).tolist()
     csv_writer.writerow(bias4)
