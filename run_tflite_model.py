import numpy as np
import cv2
import time
#load the trained model to classify sign
import tflite_runtime
import tflite_runtime.interpreter as tflite
print(np.__version__)
print(cv2.__version__)
print(tflite_runtime.__version__)
def getContours(img):
    arr = []
    x_arr = []
    y_arr = []
    key_x = True
    key_y = True
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>500:
            # cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            #print(peri)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            if objCor ==3: 
                objectType ="Tri"
            elif objCor == 4 or objCor == 5 or objCor == 6 :
                objectType="Rectangle"
            else:
                objectType="None"
            if objectType!="None":
                for i in x_arr:
                    if (abs(i-x)/x)>0.7:
                        key_x = True
                    else:
                        key_x = False
                for i in y_arr:
                    if (abs(i-y)/y)>0.7:
                        key_y = True
                    else:
                        key_y = False
                if key_x and key_y:
                    arr.append((x,y,w,h))
                    x_arr.append(x)
                    y_arr.append(y)
    return arr

def detectCircles(img):
    arr = []
    detected_circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 100, param1 = 40, param2 = 40, minRadius = 20, maxRadius = 100)
    if detected_circles is not None:
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            # Draw the circumference of the circle.
            # cv2.circle(img, (a, b), r, (0, 255, 0), 2)
            # img = cv2.rectangle(img, (a - r-10, b - r-10), (a + r+10, b + r+10), (255, 0, 0), 2)

            arr.append((a - r, b - r,2*r, 2*r))
    return arr
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        # ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    return img
def getBbox(img):
    arr = []

    # converting image into grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(gray,(7,7),1)
    imgCanny = cv2.Canny(imgBlur,90,220)
    # cv2.imshow("canny", imgCanny)
    # cv2.waitKey(1)
    cntArr = getContours(imgCanny)
    circleArr = detectCircles(imgCanny)
    arr.extend(cntArr)
    print(arr)
    arr.extend(circleArr)
    print(arr)
    return arr

interpreter = tflite.Interpreter(model_path="model_fp16.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print()
print("Input details:")
print(input_details)
print()
print("Output details:")
print(output_details)
print()
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

def detect(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (50, 50)
    # fontScale
    fontScale = 0.5
    # Blue color in BGR
    color = (255, 0, 0)
    thickness = 1

    frame = frame
    t1 = time.time()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    (height, width) = img.shape[:2] 
    # img_letter = img
    img_letter = letterbox(img,(720,480))
    arr = getBbox(img_letter)
    image = img_letter
    for i in arr:
        try:
            (x,y,w,h) = i
            if (x>=width/2) and (y<=height/2):
                img_pre = img_letter[y:y+w,x:x+h,:]
                img = img_pre/255
                img = cv2.resize(img,(32,32))
                img = img.astype(np.float32)
                img_pre =  np.reshape(img, [1, 32, 32, 3])
                interpreter.set_tensor(input_details[0]['index'],img_pre)
                interpreter.invoke()
                outputs = interpreter.get_tensor(output_details[0]['index'])[0]
                
                index = np.argmax(outputs)
                print(outputs[index])
                if outputs[index]>0.8:
                    sign=classes[index]
                    cv2.rectangle(img_letter,(x,y),(x+w,y+h),(0,255,0),1)
                    image = cv2.putText(img_letter,sign, (x,y), font, 
                                    fontScale, color, thickness, cv2.LINE_AA)
        except:
            print("error")
    print("FPS: ", int(1/(time.time() - t1+0.000001)))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        _,frame = cap.read()
        
        image = detect(frame=frame)
        cv2.imshow("image", image)
        cv2.waitKey(1)

