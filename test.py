import face_alignment
import cv2

capture = cv2.VideoCapture(0)
while 1:
    ret, frame = capture.read()
    if ret is False:
        break
    face_alignment.do_alignment(frame)

