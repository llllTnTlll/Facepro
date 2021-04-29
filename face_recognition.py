import numpy as np
import argparse
import imutils
import pickle_helper
import cv2
import os
import cfg_manager
import pickle


parent_directory = os.getcwd()
# 获取存储人脸检测器的face_detection_model路径
detector_path = os.path.sep.join([parent_directory, cfg_manager.read_cfg('Common', 'detector_path')])
# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([cfg_manager.read_cfg('Common', 'detector_path'), "deploy.prototxt"])
modelPath = os.path.sep.join([cfg_manager.read_cfg('Common', 'detector_path'), "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

print("[INFO] loading face recognizer...")
# 获取用于生成128维人脸向量的Torch模型存储位置
embedder_path = os.path.sep.join([parent_directory, cfg_manager.read_cfg('Common', 'embedder_path')])
embedderPath = os.path.sep.join([embedder_path, "openface_nn4.small2.v1.t7"])
# 加载Torch模型
embedder = cv2.dnn.readNetFromTorch(cfg_manager.read_cfg('Common', 'embedder_path'))

# 读取SVM模型及对应编码文件
recognizer = pickle.loads(open(r'./data/recognizer.pickle', "rb").read())
le = pickle.loads(open(r'./data/le.pickle', "rb").read())
# 从摄像头获取图像
# 转换每张图的宽度为600，保持其纵横比并读取高度
capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > 0.4:
            # compute the (x, y)-coordinates of the bounding box for the
            # face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # 过滤过小的人脸
            if fW < 20 or fH < 20:
                continue

            # 识别人脸姓名
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                             (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            # 使用训练好的SVM分类器识别人脸
            preds = recognizer.predict_proba(vec)[0]
            # 使用最高概率索引器查询标签编码器
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # draw the bounding box of the face along with the associated
            # probability
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.imshow('', image)
            cv2.waitKey(1)



