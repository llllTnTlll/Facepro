from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle_helper
import cfg_manager


def do_modeltrain():
    # 从pickle文件中加载人脸编码文件
    print("\033[1;33m[INFO] loading face embeddings...\033[0m")
    data = pickle_helper.load_pickle_from_disk(r'C:\Users\ZHIYUAN\PycharmProjects\Facepro\data\embeddings.pickle')

    # 生成人脸编码对应的标签
    print("\033[1;33m[INFO] encoding labels...\033[0m")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])
    print(labels)
    # 将标签文件写入硬盘
    pickle_helper.write_pickle_to_disk(r'C:\Users\ZHIYUAN\PycharmProjects\Facepro\data\le.pickle', le)

    # 使用从.pickle文件中加载的编码文件拟合训练支持向量机模型
    # 输出模型与编码文件
    print("\033[1;33m[INFO] training model...\033[0m")
    recognizer = SVC(C=float(cfg_manager.read_cfg('FaceEmbeddings', 'confidence')), kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)
    # 将训练完成的SVM模型写入硬盘
    pickle_helper.write_pickle_to_disk(r'C:\Users\ZHIYUAN\PycharmProjects\Facepro\data\recognizer.pickle', recognizer)


if __name__ == '__main__':
    do_modeltrain()
