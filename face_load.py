import cv2
import cfg_manager
import os
import imutils
import np
import threading
import time
from pynput.keyboard import Listener
import pickle_helper
import train_model
import face_encoding
import FacePro

run_flag: bool = False  # 线程全局运行标志
camera_shot = None  # 来自摄像头frame的浅度复制
main_box = []       # 存储camera_shot中人脸位置的列表
pressed_key = None  # 记录来自keyboard线程当前按下的按键


def main():
    # 向字典添加新的人脸
    name = input('\033[4;33menter your name first :\033[0m')
    # 判定人脸画像是否已存在
    isExists = os.path.exists('./face_directory/%s' % name)
    if isExists:
        key = input('name has already exists, would you like to cover it?(y/n) :')
        if key == 'n':
            FacePro.main()
    path = "./face_directory/%s" % name


# 键盘监听线程
class keyboardThread(threading.Thread):
    # 键盘监听线程
    def __init__(self):
        super().__init__()

    def run(self):
        with Listener(press) as listener:
            listener.join()
        while run_flag:
            return
        listener.stop()


class encodingThread(threading.Thread):
    def __init__(self, path):
        super().__init__()
        self.path = path  # 采集图片存储路径

    def run(self):
        do_encoding(self.path)


def do_encoding(path):
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
    while run_flag:
        if pressed_key == 's':


# 键盘事件监听器onlistener方法
def press(key):
    try:
        # 将当前按下的key覆盖至pressed_key
        global pressed_key
        pressed_key = key.char
        # 在按下q时将线程标志修改为False
        if pressed_key == 'q':
            global run_flag
            run_flag = False
    except AttributeError:
        return
