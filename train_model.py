from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle_helper
import pickle


# load the face embeddings
print("[INFO] loading face embeddings...")
# data = pickle_helper.load_pickle_from_disk(r'./data/embeddings.pickle')
data = pickle.loads(open(r'C:\Users\ZHIYUAN\PycharmProjects\Facepro\data\embeddings.pickle', "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# 使用从.pickle文件中加载的编码文件拟合训练支持向量机模型
# 输出模型与编码文件
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

f = open(r'C:\Users\ZHIYUAN\PycharmProjects\Facepro\data\recognizer.pickle', "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open(r'C:\Users\ZHIYUAN\PycharmProjects\Facepro\data\le.pickle', "wb")
f.write(pickle.dumps(le))
f.close()