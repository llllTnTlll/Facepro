import cv2
import cfg_manager


def do_alignment(image):
    # 加载Haar分类器
    eye_classifier = cv2.CascadeClassifier(cfg_manager.read_cfg('FaceAlignment', 'eye_classifier_path'))
    # 对传入照片进行浅度复制
    img = image.copy()

    (fH, fW) = img.shape[:2]
    eye_region = img[0:int(fH/2.1), 0:int(fW)]
    faceRects_eye = eye_classifier.detectMultiScale(eye_region, 1.1, 3, cv2.CASCADE_SCALE_IMAGE)

    if len(faceRects_eye) >= 0:
        for faceRect_eye in faceRects_eye:
            x1, y1, w1, h2 = faceRect_eye
            cv2.rectangle(eye_region, (int(x1), int(y1)), (int(x1) + int(w1), int(y1) + int(h2)), (0, 255, 0), 2)

    cv2.imshow('', eye_region)
    cv2.waitKey(10)








