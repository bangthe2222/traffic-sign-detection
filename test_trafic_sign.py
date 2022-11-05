import numpy as np
import cv2
import os
import time
#load the trained model to classify sign
from tensorflow.keras.models import load_model

model = load_model('my_model.h5')

#dictionary to label all traffic signs class.
classes = ['Cross Walk',
            'No Entry Road',      
            'Parking',       
            'Priority Road',      
            'Roundabout',    
            'STOP',      
            'One Way Road',  
            'HighWay Entrance',
            'HighWay Exit'   
]
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (50, 50)
  
# fontScale
fontScale = 1
   
# Blue color in BGR
color = (255, 0, 0)
thickness = 2
path = './Train/8/'
print(classes)
for i in os.listdir(path):

    frame = cv2.imread(path+ i)
# while(True):
#     _,frame = cap.read()
    t1 = time.time()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = img/255
    img = cv2.resize(img,(32,32))
    img = np.reshape(img, [1, 32, 32, 3])

    # print(img.shape)
    pred=model.predict_classes(img)[0]
    # print(pred)
    # print(pred)
    sign=classes[pred]
    image = cv2.putText(frame, sign, (20,20), font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow("image", frame)
    print(sign)
    print("FPS: ", 1/(time.time() - t1))
    cv2.waitKey(1)
# cv2.destroyAllWindows()

                 