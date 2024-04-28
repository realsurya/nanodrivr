import cv2
import tensorflow as tf
import numpy as np
import os
import time
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



# ------------------ PARAMS ------------------
fps = 30


# ------------------ IMPORT DATA ------------------
data = h5py.File("track_data_2.h5")
numel = (np.size(data['angles']))


# Create a figure and axis for the animation
fig, ax = plt.subplots()
window = ax.imshow(np.array(data['images'][0]), cmap='gray')  # Display the first frame initially

# Function to update the displayed image
def update(frame):
    window.set_data(np.array(data['images'][frame]))
    return window,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=numel, interval=1000/fps, blit=True)
