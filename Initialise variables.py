import csv
import numpy as np
#Loads libraries

with open("Object-Identifer-ML/Weights/weights01.csv","w") as f: #Creates CSV file
     csv_writer=csv.writer(f) #Turns into write mode
     weights01=np.random.normal(0, 1, size=(40000,20)).tolist() #Creates weights with normal distribution
     csv_writer.writerows(weights01) #Writes weights

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







"""
objects={"Bike":365,"Bottle":33,"Cat":202,"Chair":487,"Flower":210}
for i in objects:
        for n in range(objects[i]):
            n+=1
            im=Image.open(f"Object-Identifer-ML\Data\{i}\{i} ({n}).jpg")
            im=im.resize((200,200))
            im=im.convert("L")

            pixels = im.load()
            all_pixels = []
            for x in range(200):
                for y in range(200):
                    cpixel = pixels[x, y]
                    all_pixels.append(cpixel)
            with open(f"Object-Identifer-ML\DataCSV\{i}\{i} ({n}).csv","w") as f:
                csv_writer=csv.writer(f)
                csv_writer.writerow(all_pixels)
"""