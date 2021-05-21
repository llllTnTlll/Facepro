import cv2
import cfg_manager
import os
import imutils
import np
import threading
import time
import pickle_helper
import copy

thread_flag = False
face_box = None


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

            if confidence > 0.4:
                thread_flag = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                intbox = box.astype("int")
                (startX, startY, endX, endY) = intbox
                global face_box
                face_box = intbox
                cv2.rectangle(image, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)
            else:
                thread_flag = False

        cv2.imshow('', image)
        cv2.waitKey(1)


def do_encoding():
    print(face_box)


class myThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        print("thread start")
        while 1:
            global thread_flag
            if thread_flag is True:
                do_encoding()
                time.sleep(1)


if __name__ == '__main__':
    thread = myThread()
    thread.start()
    camera_tracking()
