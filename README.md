# FacePro（脸谱）

## 1.0 一切开始前
此git来自一个什么都不会的菜鸡的辅修毕设，目前还处在施工阶段，还有许多bug和纰漏，仅作学习交流使用。
## 1.1 需要用到的包
FacePro包含了许多已有的训练集和pip包，要正确的运行本程序，你需要以下清单：

### 主要的pip包
- opencv开源视觉库 ——**cv2**
 
- numpy矩阵运算 ——**np**
 
- 深度学习框架 ——**sklearn**
 
 ### 已训练的级联决策模型
 包括人脸及人眼（是否佩戴眼镜，左眼及右眼）识别
 
 **Tips: 该部分已全部包含在此git中 模型源自opencv官网**
 
 
 ## 1.2 运行前需要准备的事
 ### 重定向模型路径
  克隆本仓库后，请首先打开config文件夹中的cfg.ini配置文件，这里存储了本程序的全部本地设置，修改所需文件对应位置，使其与在你电脑上的模型路径一一对应。
```.ini
[Common]
detector_path = C:\Users\ZHIYUAN\PycharmProjects\Facepro\face_detection_model
embedder_path = C:\Users\ZHIYUAN\PycharmProjects\Facepro\face_encoding_model\openface_nn4.small2.v1.t7
face_directory = C:\Users\ZHIYUAN\PycharmProjects\Facepro\face_directory
pickle_directory = C:\Users\ZHIYUAN\PycharmProjects\Facepro\data\pickleHere
[FaceAlignment]
glasseseyes_classifier_path = C:\Users\ZHIYUAN\PycharmProjects\Facepro\eye_detection_model\haarcascade_eye_tree_eyeglasses.xml
lefteye_classifier_path = C:\Users\ZHIYUAN\PycharmProjects\Facepro\eye_detection_model\haarcascade_lefteye_2splits.xml
righteye_classifier_path = C:\Users\ZHIYUAN\PycharmProjects\Facepro\eye_detection_model\haarcascade_righteye_2splits.xml

```
完成上述操作后，请仔细查阅.gitignore文件中所描述的忽略文件，该部分文件涉及分支污染或隐私问题被本人从项目中剔除，请严格按照步骤重新配置对应文件
在pull此分支后，所有.gitignore中所涉及的文件都将不再同步，包括.jpg及本地pickle文件，请悉知。

### 配置用于存放照片数据的facedirectory文件夹
在项目根目录下创建一个新的文件夹，重命名为**facedirectory**，请**仔细检查拼写错误**，名称的设置会影响到程序运行。

此文件夹用于存放照片，如要手动添加全新人脸，请在facedirectory中新建文件夹，并以**人名（最好是英文或是拼音）命名此文件夹**，此文件夹下的所有.jpg照片将被作为该用户画像的一部分录入程序，画像的数目越多，编码速度越慢，识别越准确，请谨慎添加，避免无效数据污染画像。
### 配置用来存放各部分运行结果的data文件夹
data文件夹用于在本地保存编码和标签文件以及已拟合的分类器模型对应的pickle文件，请和上一步一样，在**本地创建空的data文件夹，在data文件中创建pickleHere文件夹**，在第一次运行时，程序会自动创建pickle
### 使用数据增强
**在data文件夹下创建enhanced文件夹用于存放额外的图像**，数据增强可以提高样本的多样性，增强识别能力，同时也会成倍降低编码效率
### 使用支持向量机
因为项目使用了SVM这种二分类算法，**请在第一次运行前，确保pickle文件中具有两个或两个以上的人脸编码文件，或是facedirectory中存在两个以上包含图片的人脸文件夹。** 这样程序才能正常构建lable标签文件，否则会产生 'classes must be greater than one' 错误。
## 1.3 使用说明
请参阅使用支持向量机的相关说明，确保拥有两个以上的照片文件夹，运行Encoding & TrainModel 功能生成pickle文件完成初次使用的相关配置
## 1.4 实验性功能
### 使用数据增强
数据增强可以用于提高数据集的复杂度，显著提高多识别率，同时也会降低编码运行速度，FacePro中以随机的形式提供数据增强，对每张输入图像加入旋转、均值滤波、高斯噪声其中任意一种效果。

**作者建议仅对样本较少的情况或主要识别目标启用数据增强。**
### 未完工的人脸对齐
目前face_alignment仍处于为未启用状态，仅保留源码，原因是通过人眼寻找倾斜角度过于困难，自带的级联检测器在佩戴眼镜时达不到很高的识别率
## 1.5 初始版本后的更新与修复
****
日期：_2021/6/18_ 

修复 
+ 修复在LoadNewFace时即使未录入任何人脸也会创建空文件夹的问题
+ 修复构建人脸blob时，减均值操作无效的问题

新增
+ 为数据增强的相关设计留白

****
日期： _2021/6/20_

修复
+ 修复了face_encoding中的一处循环逻辑错误，该错误会导致无意义的重复写入
+ 修复了人脸编码时若不足两组画像程序无法正常刹车的问题
+ 修复了读取pickle只能从固定路径读取的问题

新增
+ 初版数据增强已实装，现在你可以在用户管理中对特定用户启用数据增强，重新训练模型完成增强
+ 加入了pickle_helper全新的路径获取方法，减小代码重复度
+ 加入了os帮助类，减小代码重复度
+ 加入了全新的工具接口tool_interface，方便后续工具开发，目前还在施工中后续会有更多功能封装
