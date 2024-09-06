# Text-Detection-NN
 Nueral Network Code for Hand Written Text Detection

Make sure to change interpreter based on GPU or CPU implementation on Windows

The GPU version of the code works with a miniconda interpreter built on:
- python 3.9
- Tensorflow < 2.11
- CUDA 11.2
- cuDNN 8.1.0

Follow tensorflow instruction here : https://www.tensorflow.org/install/pip?hl=en#windows-native_1

VS code can change python interpreter using cntl+shift+p 
Then look up 'Python: Select Interpreter'

There are 3 main folders in this structure.

CPU and GPU are the programs I wrote that trained the Nueral Network using the CPU and GPU respectively.

Drawing and Guessing has a python program called main.py. If you run this program you will see a window pop up
where you can draw a letter using your mouse. Clicking the predict button will let the model take a guess at your
letter.