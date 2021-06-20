from imutils import paths
import numpy as np
import imutils
import cv2
import os
import cfg_manager
import pickle_helper
import oshelper


def do_embedding():
    """
    >>使用cfg_manager从硬盘加载配置文件
    >>读取模型
        人脸定位：deploy.prototxt
                res10_300x300_ssd_iter_140000.caffemodel
        Torch嵌入模型：openface_nn4.small2.v1.t7
    >>对读取到的人脸进行编码并写入 embeddings.pickle
    """
    # 错误标记
    flag = ''
    # 从磁盘加载detector
    print("\033[1;33m[INFO] loading face detector from \033[4;32m%s\033[0m" % cfg_manager.read_cfg('Common',
                                                                                                   'detector_path'))
    protoPath = os.path.sep.join([cfg_manager.read_cfg('Common', 'detector_path'), "deploy.prototxt"])
    modelPath = os.path.sep.join([cfg_manager.read_cfg('Common', 'detector_path'), "res10_300x300_ssd_iter_140000.caffemodel"])
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

    directory_path = cfg_manager.read_cfg('Common', 'face_directory')
    folder_names = os.listdir(directory_path)

    knownEmbeddings = []
    knownNames = []
    folderPaths = []
    total = 0
    nameNum = 0

    for folder_name in folder_names:
        # 获取每个文件夹的路径
        folderPaths.append(os.path.sep.join([directory_path, folder_name]))
        nameNum += 1
    name_code = -1
    for folderPath in folderPaths:
        imagePaths = list(paths.list_images(folderPath))
        name_code += 1
        # 获取到当前处理的文件夹名
        name = folder_names[name_code]
        # 判断此用户是否启用了数据增强
        isExists = os.path.exists('./data/enhanced/%s' % name)
        # 若启用了数据增强
        # 将增强后的文件路径一并写入代编码路径中
        if isExists:
            enhancedPaths = oshelper.dirwalker('./data/enhanced/%s' % name, ('jpg',))
            for enhancedPath in enhancedPaths:
                imagePaths.append(enhancedPath)
        for (i, imagePath) in enumerate(imagePaths):
            print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))

            # 加载图片，保持纵横比更改宽度维600
            # 取得标准化后的图像高度与宽度
            image = cv2.imread(imagePath)
            image = imutils.resize(image, width=600)
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
                    (fH, fW) = face.shape[:2]

                    # 剔除较小的人脸
                    if fW < 20 or fH < 20:
                        continue

                    # 构造blob
                    # 对人脸进行编码
                    faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                     (96, 96), (0, 0, 0), swapRB=True, crop=False)
                    embedder.setInput(faceBlob)
                    vec = embedder.forward()

                    # 将姓名写入列表
                    # 编码存储至列表
                    knownNames.append(name)
                    knownEmbeddings.append(vec.flatten())
                    total += 1

            else:
                print("\033[1;31m[ERROR]no face detected\033[0m")

    # 将编码写入pickle
    print("[INFO] serializing {} encodings...".format(total))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    pickle_helper.write_pickle_to_disk(pickle_helper.get_path()['embedding'], data)
    print("-------------------------------------")

    if nameNum < 2:
        print(knownNames)
        print('\033[1;31m[ERROR]classes must be greater than one\033[0m')
        flag = 'FaceNum_Error'

    return flag


