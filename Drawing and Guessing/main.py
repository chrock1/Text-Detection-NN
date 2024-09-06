import tkinter as tk
import drawing_func
from keras.models import load_model


model = load_model(drawing_func.model_path('emnist_trained_model.keras'))  # Replace 'my_trained_model.h5' with the path to your model file
root = tk.Tk()
app = drawing_func.DrawingApp(root, model)
root.mainloop()