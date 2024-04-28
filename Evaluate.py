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
    img = cv2.resize(img, (0,0), fx=10, fy=10) 
    
    
    # Add text overlay (Ground Truth)
    text = "TEST"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]


    cv2.putText(img, "Human Driver:", (10, 0+text_size[1]*2), font, font_scale, (0, 0, 0), font_thickness)
    cv2.putText(img, "Steering:"+str(np.round(data['angles'][frame],4)), (30, 0+text_size[1]*4), font, font_scale, (0, 0, 0), font_thickness)

    # Add Wheel overlay (Ground Truth)
    cv2.circle(img,(100,160), 50, (0,0,0), 5)
    cv2.line(img,(100,160),(int(np.round(100+50*np.cos((data['angles'][frame]/180)*np.pi - np.pi/2))),
                                        int(np.round(160+50*np.sin((data['angles'][frame]/180)*np.pi - np.pi/2)))),(0,0,0),5)
    

    cv2.imshow("image", img)

    time.sleep(1 / fps)
    frame += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
data.close()
cv2.destroyAllWindows()