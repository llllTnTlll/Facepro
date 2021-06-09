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
camera_shot = None     # 浅度复制capture捕获内容用于人脸编码
main_box = None        # 人脸编码所传入的主要人脸位置
pic_num = 0            # 记录共有多少张照片被捕获
capture_role = 'no_role'    # 捕获规则
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

        # 根据box面积将人脸主体与其他人分离
        # 使用绿色box标记主体，红色box标记路人

        box_list = []    # box区域列表
        box_measure = []    # box面积列表

        # 扫描循环
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            global thread_flag
            global face_box
            global camera_shot

            lock.acquire()
            if confidence > 0.4:
                thread_flag = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                intbox = box.astype("int")
                (startX, startY, endX, endY) = intbox
                face_box = intbox
                camera_shot = image.copy()
                measure = (endX - startX) * (endY - startY)    # 计算box面积
                box_measure.append(measure)    # 写入box面积列表
                box_item = [startX, startY, endX, endY]
                box_list.append(box_item)    # 写入box区域列表

            else:
                thread_flag = False
            lock.release()
        try:
            max_measure = box_measure.index(max(box_measure))
            print(box_list)
            print(box_measure)
        except ValueError:
            continue

        # 绘制循环
        for i in range(0, len(box_measure)):
            startX = box_list[i][0]
            startY = box_list[i][1]
            endX = box_list[i][2]
            endY = box_list[i][3]
            if i == max_measure:
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            else:
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, 'captured:' + str(pic_num), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('camera', image)
        cv2.waitKey(1)
    capture.release()


def do_encoding(name):
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
        pic_name = time.strftime("%Y%m%d_%H_%M_%S.jpg", time.localtime())
        cv2.imwrite("./face_directory/%s/%s" % (name, pic_name), camera_shot)    # 暂存到缓存区
        pic_num += 1    # 已捕获照片数量+1

    elif pressed_key == 'p':
        return
    elif pressed_key == 'q':
        global run_flag
        run_flag = False
    cv2.waitKey(1)


class encodingThread(threading.Thread):
    # 编码线程
    def __init__(self, isexists, name):
        threading.Thread.__init__(self)
        self.isExists = isexists
        self.name = name
        global run_flag
        run_flag = True

    def run(self):
        print("encoding thread start")
        if self.isExists:
            key = input('\033[4;33mthis name has already exists,would you like to rebuild? (y/n)\033[0m')
            if key == 'y':
                # 从硬盘中加载pickle，pop该用户的faceembedding & name
                # 重新序列化pickle
                pass
            elif key == 'n':
                return
        else:
            os.makedirs(".\\face_directory\\%s" % self.name)
            thread2 = cameraThread()
            thread3 = keyboardThread()
            thread2.start()
            thread3.start()
            global run_flag
            while run_flag:
                global thread_flag
                if thread_flag is True:
                    if pressed_key != 'q':
                        do_encoding(self.name)
                        print(pressed_key)
                        time.sleep(1)
                    else:
                        # 使用全局标识结束所有相关线程
                        run_flag = False
            cv2.destroyAllWindows()


class cameraThread(threading.Thread):
    # 摄像头捕获线程
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        print("camera thread start")
        camera_tracking()


class keyboardThread(threading.Thread):
    # 键盘监听线程
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
