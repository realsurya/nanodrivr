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

os.system("clear")

# ------------------ PARAMS ------------------
fps = 30
numHidden = 10

# ------------------ IMPORT DATA ------------------
data = h5py.File("track_data_2.h5")
numel = (np.size(data['angles']))
fw = np.shape(np.array(data['images'][0]))[0]
fh = np.shape(np.array(data['images'][0]))[1]

# ------------------ MODEL ------------------

mdl = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(fw*fh,)),  # Input layer
    tf.keras.layers.Dense(numHidden, activation='sigmoid'),  # Hidden layer
    tf.keras.layers.Dense(1)  # Output layer
])
