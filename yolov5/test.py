import tkinter as tk
from tkinter import ttk
import cv2
import torch
from PIL import Image, ImageTk
import time
import os


custom_model_path = os.path.join("models", "znaki21.pt")
if not os.path.exists(custom_model_path):
    print("Custom model file not found in the 'models' folder.")
    exit(1)


model = torch.load("znaki21.pt")
model.eval()

def initialize_webcam():
    global cap, selected_webcam
    if cap is not None:
        cap.release()  # Release the previously opened webcam

    # Get the selected webcam index from the dropdown
    selected_webcam = webcam_var.get()

    # Initialize the selected webcam
    cap = cv2.VideoCapture(selected_webcam)
selected_webcam = 0;

try:
    cap = cv2.VideoCapture(0)  # Adjust the camera index as needed
    if not cap.isOpened():
        raise Exception("Webcam capture object could not be opened.")
except Exception as e:
    print(f"Error: {e}")


root = tk.Tk()
root.title("Real-time Object Detection")


label = ttk.Label(root)
label.pack()


fps_label = ttk.Label(root, text="FPS: ")
fps_label.pack()


webcam_label = ttk.Label(root, text="Select Webcam:")
webcam_label.pack()


available_webcams = [f"Webcam {i}" for i in range(10)]  # You can adjust the range as needed


webcam_var = tk.StringVar()
webcam_var.set(selected_webcam)  # Set the default webcam


webcam_dropdown = ttk.Combobox(root, textvariable=webcam_var, values=available_webcams)
webcam_dropdown.pack()


initialize_button = ttk.Button(root, text="Initialize Webcam", command=initialize_webcam)
initialize_button.pack()


detected_objects_listbox = tk.Listbox(root, height=20, width=30)
detected_objects_listbox.pack()

frame_count = 0
start_time = time.time()

def update_frame():
    global frame_count, start_time
    ret, frame = cap.read()

    
    results = model(frame)

   
    detections = results.pred[0]

    
    annotated_frame = detections.render()[0]

    
    frame_count += 1
    if frame_count % 10 == 0:
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        fps_label.config(text=f"FPS: {fps:.2f}")

    
    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    frame_tk = ImageTk.PhotoImage(Image.fromarray(frame_rgb))

    label.imgtk = frame_tk
    label.configure(image=frame_tk)

    
    detected_objects_listbox.delete(0, tk.END)  # Clear the existing items
    for obj in detections.names:
        detected_objects_listbox.insert(tk.END, obj)

    label.after(10, update_frame)  # Update every 10 milliseconds


update_frame()


root.mainloop()


if cap is not None:
    cap.release()
cv2.destroyAllWindows()