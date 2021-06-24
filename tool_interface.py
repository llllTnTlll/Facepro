import cv2
import cfg_manager
import imutils
import numpy as np
import os


def where_face(img):

    flag = True    # 成功执行的标志
    detections = []

    protoPath = os.path.sep.join([cfg_manager.read_cfg('Common', 'detector_path'), "deploy.prototxt"])
    modelPath = os.path.sep.join(
        [cfg_manager.read_cfg('Common', 'detector_path'), "res10_300x300_ssd_iter_140000.caffemodel"])

    # 加载图片，保持纵横比更改宽度维600
    # 取得标准化后的图像高度与宽度
    image = imutils.resize(img, width=600)
    (h, w) = image.shape[:2]

    # 取得图片的RGB均值
    resized = cv2.resize(image, (300, 300))
    r_mean = np.mean(resized[:, :, 0])
    g_mean = np.mean(resized[:, :, 1])
    b_mean = np.mean(resized[:, :, 2])
    rgb_means = (r_mean, g_mean, b_mean)

    # 对整张图片构建blob
    imageBlob = cv2.dnn.blobFromImage(
        resized, 1.0, (300, 300),
        rgb_means, swapRB=False, crop=False)

    # 输入深度学习模型并取得预测结果
    try:
        detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        detector.setInput(imageBlob)
        detections = detector.forward()
    except SyntaxError:
        flag = False
    finally:
        return flag, detections


def match(img):
    pass


def photo_sticker(img):

    flag = True
    face = None
    protoPath = os.path.sep.join([cfg_manager.read_cfg('Common', 'detector_path'), "deploy.prototxt"])
    modelPath = os.path.sep.join(
        [cfg_manager.read_cfg('Common', 'detector_path'), "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    image = imutils.resize(img, width=600)
    (h, w) = image.shape[:2]

    # 取得图片的RGB均值
    resized = cv2.resize(image, (300, 300))
    r_mean = np.mean(resized[:, :, 0])
    g_mean = np.mean(resized[:, :, 1])
    b_mean = np.mean(resized[:, :, 2])
    rgb_means = (r_mean, g_mean, b_mean)

    # 对整张图片构建blob
    imageBlob = cv2.dnn.blobFromImage(
        resized, 1.0, (300, 300),
        rgb_means, swapRB=False, crop=False)

    # 输入深度学习模型并取得预测结果
    detector.setInput(imageBlob)
    detections = detector.forward()
    # 确认检测到人脸
    if len(detections) > 0:
        # 取人脸最大可能性的位置
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        # 如果置信度大于设定值
        if confidence >= float(cfg_manager.read_cfg('FaceEmbeddings', 'confidence')):
            # 取得ROI区域
            roi = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = roi.astype("int")

            # 将ROI区域截出
            face = image[startY:endY, startX:endX]
    else:
        flag = False

    return flag, face
