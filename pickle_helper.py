import pickle


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
