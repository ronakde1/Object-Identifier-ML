from PIL import Image
import csv
objects={"Bike":200,"Bottle":166,"Cat":200,"Chair":200,"Flower":200} #Size of each class
for i in objects: #For each object type
        for n in range(objects[i]): #For each object 
            n+=1 #Adds one so index starts at 1
            im=Image.open(f"Object-Identifier-ML\Data\{i}\{i} ({n}).jpg") #Calls image location
            im=im.resize((200,200)) #Resizes image
            im=im.convert("L") #Turns image greyscale

            pixels = im.load() #Loads image into pixels
            all_pixels = [] 
            for x in range(200):
                for y in range(200):
                    cpixel = pixels[x, y] #Stores pixel values in structured data type
                    all_pixels.append(cpixel)
            with open(f"Object-Identifier-ML\DataCSV\{i}\{i} ({n}).csv","w") as f:
                csv_writer=csv.writer(f)
                csv_writer.writerow(all_pixels) #Writes pixel values

