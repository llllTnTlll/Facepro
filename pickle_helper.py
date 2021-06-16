import pickle
import face_encoding


def write_pickle_to_disk(save_path, data):
    print('[INFO] save data to pickle')
    f = open(r"%s" % save_path, "wb")
    f.write(pickle.dumps(data))
    f.close()


def load_pickle_from_disk(load_path):
    print('[INFO] loading pickle from disk')
    file = open(r"%s" % load_path, "rb")
    f = pickle.load(file)
    return f


def rebuild_pickle():
    success = True
    # 尝试从硬盘中读取pickle
    try:
        data = load_pickle_from_disk(
            r'C:\Users\ZHIYUAN\PycharmProjects\Facepro\data\pickleHere\embeddings.pickle')
    # 捕获异常
    # 尝试通过已有图像重新生成pickle
    except FileNotFoundError:
        flag = face_encoding.do_embedding()
        if flag == 'FaceNum_Error':
            # 程序刹车
            success = False
        else:
            # 成功重新生成则再次读取pickle
            data = load_pickle_from_disk(
                r'C:\Users\ZHIYUAN\PycharmProjects\Facepro\data\pickleHere\embeddings.pickle')
    return success
