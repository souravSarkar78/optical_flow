''' -----------------
    Left mouse Click To select object 
    Click middle mouse button to clear selection 
'''



import cv2
import numpy as np
from numpy.core.fromnumeric import cumprod

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

        

        old_gray = new_gray.copy()    
        old_pts = new_pts.copy()

    cv2.imshow("Window", frame)

    if cv2.waitKey(1) == ord('q'):
        old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        break
        
cv2.destroyAllWindows()
cam.release()