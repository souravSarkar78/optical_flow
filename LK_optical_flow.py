import cv2
import numpy as np

ix, iy, k = 200, 200, 1
def mouseEvnt(event, x, y, flag ,param):
    global ix, iy, k
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy, k = x, y, -1
        print("Clicked")
        
cv2.namedWindow("Window")
cv2.setMouseCallback("Window", mouseEvnt)

cam = cv2.VideoCapture(0)


while True:
    _, frame = cam.read()
    cv2.imshow("Window", frame)
    if cv2.waitKey(1) == ord('q') or k== -1:
        old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.destroyAllWindows()
        break
        
mask = np.zeros_like(frame)
 
old_pts = np.array([[ix, iy]], dtype="float32").reshape(-1, 1, 2)

while True:
    _, frame2 = cam.read()
    new_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    new_pts, status, err = cv2.calcOpticalFlowPyrLK(old_gray,
                            new_gray,
                            old_pts,
                            None,
                            maxLevel=4,
                            criteria=(cv2.TERM_CRITERIA_EPS |cv2.TERM_CRITERIA_COUNT,
                                     15, 0.08)
                                                   )
                                                   
    cv2.circle(mask, (int(new_pts.ravel()[0]), int(new_pts.ravel()[1])), 2, (0, 255, 0), 2)
    frame2 = cv2.addWeighted(frame2, 1, mask, 1, 0.1)
    cv2.imshow("frame2", frame2)
    cv2.imshow("mask", mask)
    #print(( new_pts.ravel()[0], new_pts.ravel()[1]), ix, iy)
    old_gray = new_gray.copy()
    old_pts = new_pts.copy()
    if cv2.waitKey(1) == ord('q'):
        cam.release()
        cv2.destroyAllWindows()
        break