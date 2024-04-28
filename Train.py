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
numHidden = 128
alpha = 1e-4
epochs = 100
batchSize = 64
split=.7


# ------------------ IMPORT DATA ------------------
data = h5py.File("track_data_2.h5")
numel = (np.size(data['angles']))
fw = np.shape(np.array(data['images'][0]))[0]
fh = np.shape(np.array(data['images'][0]))[1]

nbins = np.size(data['encoded_angles'][0])
bins = np.linspace(-165, 165, nbins)

# ------------------ PROCESS DATA ------------------

inp = []
tgt = []

inpV = []
tgtV = []

frame = 0
while frame < np.round(split*numel):
    # image Preprocessing
    inp.append( np.reshape(np.ndarray.flatten(np.array(data['images'][frame])/255),(1, fw*fh))  )
    tgt.append(np.reshape(np.array(data['encoded_angles'][frame]), (1,nbins)))
    frame+=1
    
while frame < numel:
    # image Preprocessing
    inpV.append( np.reshape(np.ndarray.flatten(np.array(data['images'][frame])/255),(1, fw*fh))  )
    tgtV.append( np.reshape(np.array(data['encoded_angles'][frame]), (1,nbins)) )
    frame+=1

#data.close()

# ------------------ MODEL ------------------

mdl = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(fw*fh,)),  # Input layer
    tf.keras.layers.Dense(numHidden, activation='sigmoid'),  # Hidden layer
    tf.keras.layers.Dense(nbins)  # Output layer
])
mdl.summary()


mdl.compile(optimizer=tf.keras.optimizers.Adam(alpha),
              loss='mse')
mdl.fit(inp, tgt, epochs=epochs, validation_data=(inpV, tgtV), batch_size=batchSize, shuffle=True)
#mdl.fit(inp, tgt, epochs=epochs, batch_size=batchSize)
mdl.save("Models/mdl" + str(numHidden) + ".keras")

# ---------- MODEL INFERENCE ----------

frame = 0
angles = []

fr = np.reshape(np.ndarray.flatten(np.array(data['images'])/255),(numel, fw*fh))
y = mdl.predict(fr)

while frame < np.round(numel):
    # image Preprocessing
    yhat = y[frame]
    angles.append(bins[yhat.argmax(axis=0)])
    
    frame+=1


# ---------- MODEL FIT EVAL ----------
plt.figure()
plt.plot(np.array(data['angles']))
plt.plot(np.array(angles))
plt.xlabel("Index (Time)")
plt.ylabel("Steering Angle (Deg)")
plt.grid()
plt.legend(["Ground Truth (Human)", "Model Inference"])
plt.title("Driving Dataset - Model")
