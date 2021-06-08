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
run_flag = True        # 线程全局运行标志
face_box = None        # 检测到的人脸位置
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

    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while run_flag:
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
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            else:
                thread_flag = False
            lock.release()
        cv2.putText(image, 'captured:' + str(pic_num), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('camera', image)
        cv2.waitKey(1)
    capture.release()


def do_encoding():
    lock.acquire()
    print(face_box)
    global camera_shot
    if camera_shot is None:
        return
    lock.release()
    cv2.imshow('camera_shot', camera_shot)
    global pic_num
    # 若按下的按钮为s
    if pressed_key == 's':
        cv2.imwrite("./data/picSave/photo_{}.jpg".format(pic_num), camera_shot)    # 暂存到缓存区
        pic_num += 1    # 已捕获照片数量+1
        print(camera_shot)
    elif pressed_key == 'p':
        return
    elif pressed_key == 'q':
        global run_flag
        run_flag = False
    cv2.waitKey(1)


class encodingThread(threading.Thread):
    def __init__(self, isexists):
        threading.Thread.__init__(self)
        self.isExists = isexists
        global run_flag
        run_flag = True

    def run(self):
        print("encoding thread start")
        if self.isExists:
            print('this name is already exists')
        else:
            thread2 = cameraThread()
            thread3 = keyboardThread()
            thread2.start()
            thread3.start()
            global run_flag
            while run_flag:
                global thread_flag
                if thread_flag is True:
                    if pressed_key != 'q':
                        do_encoding()
                        print(pressed_key)
                        time.sleep(1)
                    else:
                        # 使用全局标识结束所有相关线程
                        run_flag = False
            cv2.destroyAllWindows()


class cameraThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        print("camera thread start")
        camera_tracking()


class keyboardThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        with Listener(press) as listener:
            listener.join()
        while run_flag:
            return
        listener.stop()


def press(key):
    global pressed_key
    try:
        pressed_key = key.char
    except AttributeError:
        return
