''' -----------------
    Left mouse Click To select object 
    Click middle mouse button to clear selection 
'''



import cv2
import numpy as np
from numpy.core.fromnumeric import cumprod

maxCropX = 150
maxCropY = 150

ix, iy, k = 0, 0, 0
old_pts = np.array([[ix, iy]], dtype="float32").reshape(-1, 1, 2)
once = 0
def mouseEvnt(event, x, y, flag ,param):
    global ix, iy, k,once,  old_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        ix = x
        iy = y
        k = 1
        once = 0
        
        print("Clicked", ix, iy)

    if event == cv2.EVENT_MBUTTONDOWN:
        k = 0
        once = 0
        
cv2.namedWindow("Window")
cv2.setMouseCallback("Window", mouseEvnt)

cam = cv2.VideoCapture(0)

width  = cam.get(3)
height = cam.get(4)

print(height, width)

while True:
    _, frame = cam.read()
    

    # Checking Condition if mouse is click for tracking or not
    if k == 1:
        
        if once == 0:
            old_pts = np.array([[ix, iy]], dtype="float32").reshape(-1, 1, 2)
            old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            once = 1

        new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        new_pts, status, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                new_gray,
                                old_pts,
                                None,
                                maxLevel=4,
                                criteria=(cv2.TERM_CRITERIA_EPS |cv2.TERM_CRITERIA_COUNT,
                                        15, 0.08)
                                                    )

        current_points = (int(new_pts.ravel()[0]), int(new_pts.ravel()[1]))
        cv2.circle(frame, current_points, 20, (0, 255, 0), 2)

        maxCropX1 = 150
        maxCropX2 = 150
        maxCropY1 = 150
        maxCropY2 = 150
        maxCropX_offset = 0
        maxCropY_offset = 0

        # Correcting X axis cropping value

        if (width - current_points[0]) <=maxCropX2:
            maxCropX2 = int((width - current_points[0]))
            maxCropX1 = int(maxCropX1 + (maxCropX1-maxCropX2))

        elif (current_points[0]) <=maxCropX1:
            maxCropX1 = int(current_points[0])
            maxCropX2 = int(maxCropX2 + (maxCropX2 - current_points[0]))


        if (height - current_points[1]) <= maxCropY2:
            maxCropY2 = int((height - current_points[1]))
            maxCropY1 = int(maxCropY1 + (maxCropY1-maxCropY2))
        elif (current_points[1]) <=maxCropY1:
            maxCropY1 = int(current_points[1])
            maxCropY2 = int(maxCropY2 + (maxCropY2 - current_points[1]))

        
        cropimg = frame[current_points[1]-(maxCropY1) : current_points[1]+(maxCropY2), current_points[0]-maxCropX1 :  current_points[0]+maxCropX2]

        #print(current_points , cropimg.shape, maxCropY_offset, maxCropY-maxCropY_offset)

        try: 
            cv2.imshow("cropimg", cropimg)
        except:
            cv2.destroyWindow("cropimg")
            k = 0

        old_gray = new_gray.copy()    
        old_pts = new_pts.copy()

    cv2.imshow("Window", frame)

    if cv2.waitKey(1) == ord('q'):
        old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        break
        
cv2.destroyAllWindows()
cam.release()