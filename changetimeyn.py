#Information about object detection
print("This program allows you to change whether you would like to activate the pause between the object classifications")
      
while True:
    #Input
    inputvalue=input("Enter a value of either Y for yes or N for no \n")
    #Validation
    if "Y" in inputvalue or "y" in inputvalue:
        finalvalue="Y"
        break
    if "N" in inputvalue or "n" in inputvalue:
        finalvalue="N"
        break
    else:
        print("not an option")
with open("Object-Identifier-ML\generalvariables\generalvariable2.txt", 'w') as file: #Opens file in write mode
    file.write(str(finalvalue)) #Writes value into file over the original value]


