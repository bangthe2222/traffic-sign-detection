import cv2
import numpy as np

def getContours(img):
    arr = []
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
            elif objCor == 4:
                aspRatio = w/float(h)
                if aspRatio >0.98 and aspRatio <1.03: objectType= "Square"
                else:objectType="Rectangle"
            elif objCor == 6:
                objectType="Hextangle"
            else:
                objectType="None"
            if objectType!="None":
                arr.append((x,y,w,h))

    return arr

def detectCircles(img):
    arr = []
    detected_circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 40, param1 = 50, param2 = 35, minRadius = 20, maxRadius = 70)
    if detected_circles is not None:

        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            # Draw the circumference of the circle.
            cv2.circle(img, (a, b), r, (0, 255, 0), 2)
            # img = cv2.rectangle(img, (a - r-10, b - r-10), (a + r+10, b + r+10), (255, 0, 0), 2)
            arr.append((a - r-10, b - r-10,2*r+10, 2*r+10))
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
    # img = np.ascontiguousarray(img)
    return img
def getBbox(img):
    # reading image
    # img = cv2.imread('images.png')
    arr = []
    width = 720
    height = 480
    dim = (width, height)
    # img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img = letterbox(img,(height,width))

    # converting image into grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur using 3 * 3 kernel.
    # gray_blurred = cv2.blur(gray, (3, 3))

    # Apply Hough transform on the blurred image.




    # imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(gray,(7,7),1)
    imgCanny = cv2.Canny(imgBlur,50,50)
    cntArr = getContours(imgCanny)
    circleArr = detectCircles(imgCanny)
    arr.extend(cntArr)
    arr.extend(circleArr)
    return arr