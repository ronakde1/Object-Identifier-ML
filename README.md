# Object-Identifier-ML
## Set up guide
### Hardware required for setup
•	Raspberry Pi with cooler

•	SD card 32Gb

•	Computer with SD card slot

•	Monitor

•	Keyboard and mouse

•	Internet connection 

First, install the Raspberry Pi Imager from this website: https://www.raspberrypi.com/software/ 

Then, run program and install the correct Raspberry Pi operating system. It is important to install 64 bit if the Raspberry Pi uses this and 32 bit otherwise. A 64 bit operating system is not compatible with a 32 bit Raspberry Pi and could potentially cause problems. 

Then select SD card from the choose storage option. Ensure the correct drive is chosen because it will be wiped when run. 

Then select write. Ater this has been completed, place micro-SD card within Raspberry Pi. After this point, there are 2 ways to continue. This is one of the ways. The other way includes opening a SSH session on the main computer using WSL and connecting to the Raspberry Pi from this. Programs can then be opened using the commands to open the executable locations. Since this is a headless installation methodology, the user must be competent using commands to use this method so it is not recommended.

Then attach monitor, keyboard and mouse to Raspberry Pi using the USB ports and the mini HDMI connection. 

Then connect to power. Raspberry Pi should boot. If not, repeat the first steps and use a different SD card. There are many reasons on why the Raspberry Pi does not boot. Check the error message  on the Raspberry Pi and compare to documentation. 

Once Raspberry Pi is booted, enter username and password to program. Then connect device to internet. 

After this, install Python if not done already and install Git on the Raspberry Pi. Visual studio code is also required to be installed. You must also sign into visual studio code on the computer. This can be done by opening a git terminal and running the following commands with your email and name 
 
You can check the git configuration files using the command below 
 
After this, go to visual studio code and type ```ctrl shift p```. Then select pull and enter URL of git repository. If the program asks you to store the program in a specific location, ensure that the folder name is ```Object-Identifier-ML```. This is not the same as the repository so this must be changed. If this is not possible, this can be changed within the programs. 

After this, set up a virtual environment and locally install the libraries. You will know which libraries to install depending on if the libraries are underlined. To install a specific library within the virtual environment, do the command ```pip install <library-name>```. If this doesn't work, then a global installation of the library may be done, if not optimal by opening a terminal and typing```python3 pip install <library-name>```. 

After this, run the programs in the following order. If not done, in this order, this could potentially cause the program to give an error. 

1.	Run preprocessing file. This will produce the CSV files from the images.
2.	Run initialise variables. This will generate the random weights and biases.
3.	Run Neural Network training. This will train the network for that set of images. This will take approximately 5 hours of running on a powerful computer. This cannot be run on a Raspberry Pi in a reasonable time frame, with each of the 400 mini-epoch cycles taking around 15 minutes and therefore requries the program to be run on an EC2 instance. This can be done from the Raspberry Pi relativley easily however will cost around £30 to fully train using the EC2 instance. 
4.	Run change pitch program.
5.	Run neural network after training.

The program should now running as expected. If the program as a low accuracy, it can easily be retrained by running the “initialise variables” file again and running the Neural Network training program. 
