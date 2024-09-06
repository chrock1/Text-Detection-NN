import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageGrab
import numpy as np
from keras.models import load_model
import os

current_directory = os.getcwd()
current_directory = os.path.join(current_directory,'CPU')

def model_path(file_name):
    path = os.path.join(current_directory, 'Models',file_name)
    return path


class DrawingApp:
    def __init__(self, root, model):
        self.root = root
        self.root.title("Draw in the Square")

        self.canvas = tk.Canvas(root, width=400, height=400, bg="black")
        self.canvas.pack()

        #Frame for buttons
        button_frame = tk.Frame(root)
        button_frame.pack(side=tk.BOTTOM, pady=10)  # Pack the frame at the bottom with padding

        # Predict button
        self.predict_button = tk.Button(button_frame, text="Predict Letter", command=self.predict_letter)
        self.predict_button.pack(side=tk.LEFT, padx=5)  # Pack to the left within the button frame

        #Clear Button
        self.clear_button = tk.Button(button_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=5)  # Pack to the left within the button frame

        #Bind mouse movement
        self.canvas.bind("<B1-Motion>", self.draw)
        #bind mouse press
        self.canvas.bind("<ButtonPress-1>", self.start_draw) 

        #This is the square drawing box for the user
        self.canvas.create_rectangle(100, 100, 300, 300, outline="white", width=2)

        #These variables will keep track of the last position of the mouse so we can connect the lines
        self.last_x, self.last_y = None, None

        #Store the model passed as an argument
        self.model = model

 

    def start_draw(self, event):
        #Stre the starting position of the mouse
        self.last_x, self.last_y = event.x, event.y

 

    def draw(self, event):
        #Get the current mouse position
        x, y = event.x, event.y

        #Draw only if the mouse is within the square box
        if 100 <= x <= 300 and 100 <= y <= 300:
            if self.last_x is not None and self.last_y is not None:
                #Draw line from start position to new position
                self.canvas.create_line(self.last_x, self.last_y, x, y, fill="white", width=5, smooth=True, splinesteps=36) #This can take some tweaking
                                                                                                                            #You can tweak the width and smoothing to try and
                                                                                                                            #get a nice thick line that has no gaps
            self.last_x, self.last_y = x, y

        else:
            #Reset last position when the mouse leaves the square
            self.last_x, self.last_y = None, None

    def clear_canvas(self):
        #Clear canvas
        self.canvas.delete("all")

        #Redraw the square box
        self.canvas.create_rectangle(100, 100, 300, 300, outline="white", width=2)

 

    def predict_letter(self):
        self.canvas.update()
        x = self.root.winfo_rootx() + self.canvas.winfo_x() + 100
        y = self.root.winfo_rooty() + self.canvas.winfo_y() + 100

        x1 = x + 200
        y1 = y + 200

        #Grab the image from the canvas
        image = ImageGrab.grab().crop((x, y, x1, y1))

        #Convert the image to grayscale and resize it to 28x28 pixels
        image = image.convert('L').resize((28, 28), Image.LANCZOS)

        #We should be normalizng the pixel values but it looks like the NN was not
        #trained on normalized data so unormalized actually works way better. I might go back
        #and train the NN on normalized data.
        image_data = np.array(image)# / 255.0

        #Match EMINST dimensions
        image_data = np.expand_dims(image_data, axis=-1)  # Add channel dimension for grayscale
        image_data = np.expand_dims(image_data, axis=0)  # Add batch dimension

        #Get the prediction
        prediction = self.model.predict(image_data)
        predicted_letter = np.argmax(prediction)

        print(f"{chr(predicted_letter + 65)}")
        #Show the prediction in a dialog window
        messagebox.showinfo(" #Prediction", f"Predicted letter: Test #{chr(predicted_letter + 65)}")  # Adjust as needed for EMNIST label mapping
