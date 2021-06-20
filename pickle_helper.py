import pickle
import face_encoding
import os
import cfg_manager


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
    pickledic = get_path()
    success = True
    # 尝试从硬盘中读取pickle
    try:
        data = load_pickle_from_disk(pickledic['embedding'])
    # 捕获异常
    # 尝试通过已有图像重新生成pickle
    except FileNotFoundError:
        flag = face_encoding.do_embedding()
        if flag == 'FaceNum_Error':
            # 程序刹车
            success = False
        else:
            # 成功重新生成则再次读取pickle
            data = load_pickle_from_disk(pickledic['embedding'])
    return success


def get_path():
    pickle_directory = cfg_manager.read_cfg('Common', 'pickle_directory')
    empickle = os.path.sep.join([pickle_directory, "embeddings.pickle"])
    lepickle = os.path.sep.join([pickle_directory, "le.pickle"])
    recopickle = os.path.sep.join([pickle_directory, "recognizer.pickle"])
    pickledic = {'embedding': empickle, 'le': lepickle, 'recognizer': recopickle}
    return pickledic

