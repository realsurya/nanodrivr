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


# ------------------ IMPORT DATA ------------------
data = h5py.File("track_data_2.h5")
numel = (np.size(data['angles']))

# ------------------ PLAYBACK DATA ------------------


frame = 0
while frame < numel:
    
    # image Preprocessing
    img = cv2.cvtColor(np.stack((data['images'][frame],) * 3, axis=-1), cv2.COLOR_BGR2GRAY)
    
    cv2.imshow("image", img)

    time.sleep(1 / fps)
    frame += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
data.close()
cv2.destroyAllWindows()