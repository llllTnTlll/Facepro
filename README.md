# 毕业设计项目——Facepro
## 1.1 需要用到的包
Facepro包含了许多已有的训练集和pip包，要正确的运行本程序，你需要以下清单：

### 主要的pip包
- opencv开源视觉库 ——**cv2**
 
- numpy矩阵运算 ——**np**
 
- 深度学习框架 ——**sklearn**
 
 ### 已训练的级联决策模型
 包括人脸及人眼（是否佩戴眼镜，左眼及右眼）识别
 
 **Tips: 该部分已全部包含在此git中 模型源自opencv官网**
 
 
 ## 1.2 运行前需要准备的事
  克隆本仓库后，请首先打开config文件夹中的cfg.ini配置文件，这里存储了本程序的全部本地设置，修改所需文件对应位置，使其与在你电脑上的模型路径一一对应。
```.ini
[Common]
detector_path = C:\Users\ZHIYUAN\PycharmProjects\Facepro\face_detection_model
embedder_path = C:\Users\ZHIYUAN\PycharmProjects\Facepro\face_encoding_model\openface_nn4.small2.v1.t7
face_directory = C:\Users\ZHIYUAN\PycharmProjects\Facepro\face_directory
[FaceAlignment]
glasseseyes_classifier_path = C:\Users\ZHIYUAN\PycharmProjects\Facepro\eye_detection_model\haarcascade_eye_tree_eyeglasses.xml
lefteye_classifier_path = C:\Users\ZHIYUAN\PycharmProjects\Facepro\eye_detection_model\haarcascade_lefteye_2splits.xml
righteye_classifier_path = C:\Users\ZHIYUAN\PycharmProjects\Facepro\eye_detection_model\haarcascade_righteye_2splits.xml

```
完成上述操作后，请仔细查阅.gitgnore文件中所描述的忽略文件，该部分文件涉及分支污染或隐私问题被本人从项目中剔除，请严格按照步骤重新配置对应文件
