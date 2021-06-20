import cv2
import os
import cfg_manager
import oshelper
import random
import tool_interface
import numpy as np

output = []


def do_enhanement(name):
    # 获取指定name的图像路径列表
    directory_path = cfg_manager.read_cfg('Common', 'face_directory')
    folderPath = os.path.sep.join([directory_path, name])
    pathlist = oshelper.dirwalker(folderPath, ('.jpg', '.jpeg'))

    ePath = './data/enhanced/%s' % name   # 数据增强存储地址
    isExists = os.path.exists(ePath)
    if isExists:
        key = input('name has already exists, would you like to cover it?(y/n) :')
        if key == 'n':
            return
        if key == 'y':
            # 删除文件夹中所有增强文件
            # 保留原有目录
            oshelper.deleteAll(path=ePath, removedir=False)
    else:
        # 若不存在创建目录
        os.makedirs('./data/enhanced/%s' % name)
    # 对指定照片库使用数据增强
    for imagePath in pathlist:
        img = cv2.imread(imagePath)
        # 随机应用多种变换中的一种
        random_key = random.randint(1, 3)
        if random_key == 1:
            rotate(img)
        elif random_key == 2:
            flip(img)
        elif random_key == 3:
            blurred(img)
    # 将增强过的列表写入文件夹
    i = 0
    for pic in output:
        pic_name = "enhanced%i.jpg" % i
        cv2.imwrite('./data/enhanced/%s/%s' % (name, pic_name), pic)  # 存至硬盘
        i += 1
    # 验证图片可用性
    # 移除无效增强
    enumerated_list = oshelper.dirwalker(ePath, ('.jpg', '.jpeg'))
    for enumerated in enumerated_list:
        enumerated_pic = cv2.imread(enumerated)
        flag, detections = tool_interface.where_face(enumerated_pic)
        if flag:
            if len(detections) > 0:
                # 取人脸最大可能性的位置
                i = np.argmax(detections[0, 0, :, 2])
                confidence = detections[0, 0, i, 2]

                # 如果置信度小于设定值
                if confidence < float(cfg_manager.read_cfg('FaceEmbeddings', 'confidence')):
                    os.remove(enumerated)
                    print('removed')
                else:
                    print('enumerated')
        else:
            print('\033[1;31m[ERROR]enumerate failed\033[0m')


# 水平翻转
def flip(img):
    global output
    h_filp = cv2.flip(img, 1)  # 水平翻转
    output.append(h_filp)


# 左右旋转15度
def rotate(img):
    global output
    angles = [15, 345]
    H, W = img.shape[:2]
    center = (H / 2, W / 2)
    for angle in angles:
        M = cv2.getRotationMatrix2D(center, angle, 1)  # 获得图像旋转矩阵
        img = cv2.warpAffine(img, M, (H, W),
                             borderValue=(255, 255, 255))
        output.append(img)


# 高斯滤波与均值滤波
def blurred(img):
    global output
    gaussian = cv2.GaussianBlur(img, (9, 9), 0)  # 高斯模糊
    blur = cv2.blur(img, (9, 9), (-1, -1))  # 均值滤波
    output.append(gaussian)
    output.append(blur)


do_enhanement('zhiyuan')