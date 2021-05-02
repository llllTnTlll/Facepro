import cv2
import cfg_manager
import operator
from functools import reduce
import np


def do_alignment(image):
    # 加载Haar分类器
    glasses_eyes_classifier = cv2.CascadeClassifier(cfg_manager.read_cfg('FaceAlignment', 'glasseseyes_classifier_path'))
    lefteye_classifier = cv2.CascadeClassifier(cfg_manager.read_cfg('FaceAlignment', 'lefteye_classifier_path'))
    righteye_classifier = cv2.CascadeClassifier(cfg_manager.read_cfg('FaceAlignment', 'righteye_classifier_path'))
    # 对传入照片进行浅度复制
    img = image.copy()

    # 获取人脸图像尺寸
    (fH, fW) = img.shape[:2]
    # 裁剪人脸上半部分
    eye_region = img[0:int(fH/2.1), 0:int(fW)]
    # 首先使用Haar分类器对两只眼睛分别进行定位
    lefteye = lefteye_classifier.detectMultiScale(eye_region, 1.2, 10, cv2.CASCADE_SCALE_IMAGE)
    righteye = righteye_classifier.detectMultiScale(eye_region, 1.2, 10, cv2.CASCADE_SCALE_IMAGE)
    # 若两个分类器均检测到结果
    if len(righteye) == 1 and len(lefteye) == 1:
        # 分析左右眼坐标差异性
        difference = reduce(operator.add, lefteye - righteye)
        var = np.var(difference)
        # 若左右眼区域之差的方差大于200
        # 视为成功分别检测到左右眼
        if var > 200:
            faceRects_eye = [lefteye, righteye]
            for faceRect_eye in faceRects_eye:
                # 压缩二维数组至一维
                _faceRect_eye = reduce(operator.add, faceRect_eye)
                x1, y1, w1, h1 = _faceRect_eye
                cv2.rectangle(eye_region, (int(x1), int(y1)), (int(x1) + int(w1), int(y1) + int(h1)), (0, 255, 0), 2)
                return _faceRect_eye

    else:
        glasses_eyes = glasses_eyes_classifier.detectMultiScale(eye_region, 1.2, 10, cv2.CASCADE_SCALE_IMAGE)

    cv2.imshow('', eye_region)
    cv2.waitKey(10)








