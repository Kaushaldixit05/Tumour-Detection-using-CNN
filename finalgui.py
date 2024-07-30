import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinter import filedialog
from frames import *
from displayTumour import *
from predictTumour import *
def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg *.gif *.bmp")])
    if file_path:
        image = Image.open(file_path)
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo
        file_path_label.config(text=f"Selected Image: {file_path}")
        y_pred = predictTumor(Image)
        if y_pred <=0.5:
            print("Does not content brain tumour and hence its Healthy ")
        else:
            print("Does content Brain tumour and hence its Cancer")
        print(y_pred)

app = tk.Tk()
app.title("Image Viewer")

# Set the initial window size (width x height)
app.geometry("800x600")

# Create a label to display the selected file path
file_path_label = tk.Label(app, text="Selected Image: None")
file_path_label.pack()

open_button = tk.Button(app, text="Open Image", command=open_image)
open_button.pack()

image_label = tk.Label(app)
image_label.pack()

app.mainloop()
