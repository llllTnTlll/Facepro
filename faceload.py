import cv2
import cfg_manager
import os
import imutils
import np
import threading
import time
from pynput.keyboard import Listener

pressed_key = None     # 监视键盘输入
thread_flag = False    # 在未检测到人脸是不进行人脸编码
face_box = None
camera_shot = None
capture_role = 'no_role'
pic_num = 0
lock = threading.Lock()


def camera_tracking():
    # 从磁盘加载detector
    print("\033[1;33m[INFO] loading face detector from \033[4;32m%s\033[0m" % cfg_manager.read_cfg('Common',
                                                                                                   'detector_path'))
    protoPath = os.path.sep.join([cfg_manager.read_cfg('Common', 'detector_path'), "deploy.prototxt"])
    modelPath = os.path.sep.join(
        [cfg_manager.read_cfg('Common', 'detector_path'), "res10_300x300_ssd_iter_140000.caffemodel"])
    try:
        detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    except cv2.error:
        # 返回模型加载失败错误
        print('\033[1;31m[ERROR]load detector failed\033[0m')
        print(">>>HELP")
        print("  * Confirm that the corresponding model file exists in the specified path")
        flag = 'Detector_Error'
        return flag
    print("\033[22;32m>>>success\033[0m")
    # 从磁盘中加载embedder
    print("\033[1;33m[INFO] loading face recognizer...\033[0m")
    try:
        embedder = cv2.dnn.readNetFromTorch(cfg_manager.read_cfg('Common', 'embedder_path'))
    except cv2.error:
        # 返回模型加载失败错误
        print('\033[1;31m[ERROR]load embedder failed\033[0m')
        print(">>>HELP")
        print("  * Confirm that the corresponding model file exists in the specified path")
        flag = 'Embedder_Error'
        return flag
    print("\033[22;32m>>>success\033[0m")

    capture = cv2.VideoCapture(0)

    while 1:
        ret, frame = capture.read()
        if ret is False:
            break
        image = imutils.resize(frame, width=600)
        (h, w) = image.shape[:2]
        # 构建Blob
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)
        detector.setInput(imageBlob)
        detections = detector.forward()

        # 提取人脸ROI
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            global thread_flag

            lock.acquire()
            if confidence > 0.4:
                thread_flag = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                intbox = box.astype("int")
                (startX, startY, endX, endY) = intbox
                global face_box
                face_box = intbox
                global camera_shot
                camera_shot = image.copy()
                cv2.rectangle(image, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
            else:
                thread_flag = False
            lock.release()
        cv2.putText(image, 'captured:' + str(pic_num), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('', image)
        cv2.waitKey(1)


def do_encoding():
    lock.acquire()
    print(face_box)
    if camera_shot is None:
        return
    lock.release()
    cv2.imshow('camera_shot', camera_shot)
    global pic_num
    pic_num += 1
    cv2.waitKey(1)


#
class encodingThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        print("encoding thread start")
        while 1:
            global thread_flag
            if thread_flag is True:
                do_encoding()
                print(pressed_key)
                time.sleep(1)


class cameraThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        print("camera thread start")
        while 1:
            camera_tracking()


class keyboardThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        with Listener(press) as listener:
            listener.join()


def press(key):
    global pressed_key
    pressed_key = key


if __name__ == '__main__':

    thread1 = encodingThread()
    thread2 = cameraThread()
    thread3 = keyboardThread()
    thread1.start()
    thread2.start()
    thread3.start()
