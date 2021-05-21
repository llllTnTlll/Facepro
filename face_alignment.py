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
    # 初始化人眼中心点数组
    eye_location = []

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
                # 标出人眼中心点
                eye_region_center = (int(x1)+int(w1/2), int(y1)+int(h1/2))
                cv2.circle(eye_region, eye_region_center, radius=3, color=(0, 0, 255))
                for num in eye_region_center:
                    eye_location.append(num)
            if len(eye_location) == 4:
                cv2.line(eye_region, (eye_location[0], eye_location[1]), (eye_location[2], eye_location[3]), (0, 0, 255), thickness=3)
                k = (eye_location[3]-eye_location[1])/(eye_location[2]-eye_location[0])

            cv2.imshow('eye', eye_region)
            cv2.waitKey(10)
            return

        else:
            print('face_alignment failed')

    # 当仅仅检测到一只眼时
    elif len(righteye) == 1 or len(lefteye) == 1:
        faceRects_eye = [lefteye, righteye]
        for faceRect_eye in faceRects_eye:
            # 若存在某一人眼的坐标
            # 压缩二维数组至一维
            if len(faceRect_eye) != 0:
                _faceRect_eye = reduce(operator.add, faceRect_eye)
                print(_faceRect_eye)
                # 使用戴眼镜的人眼检测查找另一只眼睛
                glasses_eyes = glasses_eyes_classifier.detectMultiScale(eye_region, 1.2, 10, cv2.CASCADE_SCALE_IMAGE)
                print(glasses_eyes)
                # 若能找到两只人眼
                if len(glasses_eyes) == 2:
                    # 分析左右眼坐标差异性
                    difference = glasses_eyes[0] - glasses_eyes[1]
                    var = np.var(difference)
                    # 若左右眼区域之差的方差大于200
                    # 视为成功分别检测到左右眼
                    if var > 200:
                        for glasses_eye in glasses_eyes:
                            # 压缩二维数组至一维
                            _faceRect_eye = reduce(operator.add, glasses_eye)
                            x1, y1, w1, h1 = _faceRect_eye
                            cv2.rectangle(eye_region, (int(x1), int(y1)), (int(x1) + int(w1), int(y1) + int(h1)),
                                          (0, 255, 0), 2)
                            cv2.imshow('eye', eye_region)
                            cv2.waitKey(10)
                        return
                    else:
                        print('face_alignment failed')

                # 若只检测到一只眼
                elif len(glasses_eyes) == 1:
                    eyes = [faceRect_eye, glasses_eyes]
                    difference = reduce(operator.add, eyes[0]-eyes[1])
                    var = np.var(difference)
                    if var > 200:
                        for eye in eyes:
                            _faceRect_eye = reduce(operator.add, eye)
                            x1, y1, w1, h1 = _faceRect_eye
                            cv2.rectangle(eye_region, (int(x1), int(y1)), (int(x1) + int(w1), int(y1) + int(h1)),
                                          (0, 255, 0), 2)
                            cv2.imshow('eye', eye_region)
                            cv2.waitKey(10)
                        return
                    else:
                        print('face_alignment failed')
            else:
                print('face_alignment failed')
    else:
        print('face_alignment failed')
