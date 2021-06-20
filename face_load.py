import cv2
import cfg_manager
import os
import imutils
import numpy as np
import threading
import time
from pynput.keyboard import Listener, Key
import pickle_helper
import train_model
import face_encoding

run_flag: bool = False  # 线程全局运行标志
camera_shot = None  # 来自摄像头frame的浅度复制
main_box = []       # 存储camera_shot中人脸位置的列表
pic_num = 0         # 捕获并编码的人脸数量
knownNames = []
knownEmbeddings = []
pressed_key = None  # 记录来自keyboard线程当前按下的按键


def startLoad():
    # 向字典添加新的人脸
    name = input('\033[4;33menter your name first :\033[0m')
    # 判定人脸画像是否已存在
    isExists = os.path.exists('./face_directory/%s' % name)
    if isExists:
        key = input('name has already exists, would you like to cover it?(y/n) :')
        if key == 'n':
            return
    else:
        os.makedirs('./face_directory/%s' % name)
    global run_flag
    run_flag = True
    kthread = keyboardThread()
    cthread = cameraThread()
    ethread = encodingThread(name)
    kthread.start()
    cthread.start()
    ethread.start()
    kthread.join()
    cthread.join()
    ethread.join()


# 键盘监听线程
class keyboardThread(threading.Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        with Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()


def on_press(key):
    global run_flag
    global pressed_key
    try:
        # 在按下q时将线程标志修改为False
        if key == Key.esc:
            run_flag = False
        # 将当前按下的key覆盖至pressed_key
        pressed_key = key.char
    except AttributeError:
        return


def on_release(key):
    if key == Key.esc:
        # Stop listener
        return False


class encodingThread(threading.Thread):
    def __init__(self, name):
        super().__init__()
        self.name = name  # 采集图片存储路径

    def run(self):
        do_encoding(self.name)


class cameraThread(threading.Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        camera_tracking()


def do_encoding(name):
    global run_flag
    global main_box
    global knownNames
    global knownEmbeddings
    global pic_num

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

    # 获取编码pickle路径
    pickledic = pickle_helper.get_path()
    # 尝试从硬盘中读取pickle
    try:
        data = pickle_helper.load_pickle_from_disk(pickledic['embedding'])
        knownNames = data["names"]  # pickle中已有的name
        knownEmbeddings = data["embeddings"]  # pickle中以有的人脸编码
    # 捕获异常
    # 尝试通过已有图像重新生成pickle
    except FileNotFoundError:
        flag = face_encoding.do_embedding()
        if flag == 'FaceNum_Error':
            # 程序刹车
            run_flag = False
        else:
            # 成功重新生成则再次读取pickle
            data = pickle_helper.load_pickle_from_disk(pickledic['embedding'])
            knownNames = data["names"]  # pickle中已有的name
            knownEmbeddings = data["embeddings"]  # pickle中以有的人脸编码

    pic_num = 0
    while run_flag:
        # 按下s开始编码
        if pressed_key == 's':
            # 若main_box中存在数据
            # 说明camera_shot中存在人脸需要执行编码
            if len(main_box) != 0:
                pic_name = time.strftime("%Y%m%d_%H_%M_%S.jpg", time.localtime())
                cv2.imwrite("./face_directory/%s/%s" % (name, pic_name), camera_shot)  # 存至硬盘
                startX = main_box[0]
                startY = main_box[1]
                endX = main_box[2]
                endY = main_box[3]
                # 将ROI区域截出
                face = camera_shot[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                # 剔除较小的人脸
                if fW < 20 or fH < 20:
                    continue

                # 取得图片RGB均值
                r_mean = np.mean(face[:, :, 0])
                g_mean = np.mean(face[:, :, 1])
                b_mean = np.mean(face[:, :, 2])
                rgb_means = (r_mean, g_mean, b_mean)

                # 构造blob
                # 对人脸进行编码
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), rgb_means, swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                # 将姓名写入列表
                # 编码存储至列表
                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())

                pic_num += 1  # 已捕获照片数量+1
        # 按下任意键暂停编码
        else:
            continue

        time.sleep(1)
    if pic_num != 0:
        data = {"embeddings": knownEmbeddings, "names": knownNames}
        pickle_helper.write_pickle_to_disk(pickledic['embedding'], data)
        train_model.do_modeltrain()
    else:
        os.removedirs('./face_directory/%s' % name)
        print('[INFO] no pic added')


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

    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while run_flag:
        ret, frame = capture.read()
        if ret is False:
            break
        image = imutils.resize(frame, width=600)
        (h, w) = image.shape[:2]

        # 取得图片的RGB均值
        resized = cv2.resize(image, (300, 300))
        r_mean = np.mean(resized[:, :, 0])
        g_mean = np.mean(resized[:, :, 1])
        b_mean = np.mean(resized[:, :, 2])
        rgb_means = (r_mean, g_mean, b_mean)

        # 构建Blob
        imageBlob = cv2.dnn.blobFromImage(
            resized, 1.0, (300, 300),
            rgb_means, swapRB=False, crop=False)
        detector.setInput(imageBlob)
        detections = detector.forward()

        # 根据box面积将人脸主体与其他人分离
        # 使用绿色box标记主体，红色box标记路人

        box_list = []    # box区域列表
        box_measure = []    # box面积列表

        global main_box
        global camera_shot
        # 扫描循环
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.4:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                intbox = box.astype("int")
                (startX, startY, endX, endY) = intbox
                face_box = intbox
                camera_shot = image.copy()
                measure = (endX - startX) * (endY - startY)    # 计算box面积
                box_measure.append(measure)    # 写入box面积列表
                box_item = [startX, startY, endX, endY]
                box_list.append(box_item)    # 写入box区域列表

        try:
            max_measure = box_measure.index(max(box_measure))
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
                global main_box
                main_box = [startX, startY, endX, endY]
            else:
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, 'captured:' + str(pic_num), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('camera', image)
        cv2.waitKey(1)
    capture.release()
    cv2.destroyAllWindows()
