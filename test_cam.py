# import numpy as np
# import cv2

# cap = cv2.VideoCapture(0)

# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     # # Our operations on the frame come here
#     # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = frame

#     # Display the resulting frame
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()

import numpy as np
import cv2
import pdb 

# object detection based on colors
def find_location(frame):
    idx = (frame[:, :, 0] < 30 ) & (frame[:, :, 1] < 30) & (frame[:, :, 2] > 127)
    thresh = np.uint8(idx)*255
    # indices = np.where(idx==True)
    # contours,hierarchy = cv2.findContours(idx, 1, 2)
    # pdb.set_trace()

    # imgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(imgray, 200, 255, cv2.THRESH_BINARY)
    
    # filename = 'savedImage.jpg'
    # cv2.imwrite(filename, thresh)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    else:
        return contours

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
print( (int(cap.get(3)),int(cap.get(4))))
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (int(cap.get(3)),int(cap.get(4))))


while(cap.isOpened()):
    ret, frame = cap.read()
    cnt = find_location(frame)

    if cnt is not None:
        for _cnt in cnt:
            x,y,w,h = cv2.boundingRect(_cnt)
            # pdb.set_trace()
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        
    if ret==True:
        # frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()


