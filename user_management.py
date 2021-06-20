import pickle_helper
import train_model
import cfg_manager
import oshelper
import os

selected = ''  # 已选中用户名称
directory_path = ''
folder_names = None


def do_management():
    # 得到name_list
    try:
        data = pickle_helper.load_pickle_from_disk(
            r'C:\Users\ZHIYUAN\PycharmProjects\Facepro\data\pickleHere\embeddings.pickle')
        names = data["names"]
        name_list = []
    except FileNotFoundError:
        print("\033[1;31m[ERROR]no pickle here, please load your face and encode first\033[0m")
        return
    for name in names:
        if name not in name_list:
            name_list.append(name)
    print('|  id  |      name      |')
    for i in range(len(name_list)):
        print(' %i       %s' % (i, name_list[i]))

    # 选定操作用户
    flag = True
    global selected
    while flag:
        try:
            n = int(input('\033[4;33menter an id :\033[0m'))
            selected = name_list[n]
            flag = False
        except ValueError:
            print('no such id please enter again ')
        except IndexError:
            print('index out of range')
    try:
        global directory_path
        global folder_names
        directory_path = cfg_manager.read_cfg('Common', 'face_directory')
        folder_names = os.listdir(os.path.sep.join([directory_path, selected]))
        lastchange = 0
        for folder_name in folder_names:
            time = folder_name.replace('_', '').replace('.jpg', '')
            if int(time) > int(lastchange):
                lastchange = time
    finally:
        pass

    # 展示选中用户信息
    print('\033[1;32m=======================User Info========================\033[0m')
    print('- Name: %s' % selected)
    print('- Pic num: %i' % names.count(selected))
    print('- Last modified: %s' % lastchange)
    print('\033[1;32m========================================================\033[0m')
    print('press 1 for : Delete')
    print('press 2 for : Backup')
    # 选择对用户的操作
    key = input('\033[4;33menter num and press enter : \033[0m')
    function_choose(key)


def function_choose(num):
    # 功能选择
    numbers = {
        '1': delete,
        '2': backup,

    }
    numbers.get(num, default)()


def default():
    # 在输入不属于字典的function_num时显示提示
    print('\033[1;31mError:no such function\033[0m')
    key = input('\033[4;33mpress num and press enter :\033[0m')
    function_choose(key)


def delete():
    # 加载所有已存在用户信息
    data = pickle_helper.load_pickle_from_disk(
        r'C:\Users\ZHIYUAN\PycharmProjects\Facepro\data\pickleHere\embeddings.pickle')
    knownNames = data["names"]  # pickle中已有的name
    knownEmbeddings = data["embeddings"]  # pickle中以有的人脸编码

    # 从embeddings.pickle中移除选中的用户信息
    while 1:
        try:
            index = knownNames.index(selected)
            del knownNames[index]
            del knownEmbeddings[index]
            print(index)
        except ValueError:
            break

    data = {"embeddings": knownEmbeddings, "names": knownNames}
    pickle_helper.write_pickle_to_disk(r"C:\Users\ZHIYUAN\PycharmProjects\Facepro\data\pickleHere\embeddings.pickle",
                                       data)

    delDir = os.path.sep.join([directory_path, selected])
    # 彻底清理本地照片及目录
    oshelper.deleteAll(delDir, True)
    # 重新训练模型
    train_model.do_modeltrain()


def data_augmentation():
    print("\033[1;33m[INFO] Data enhancement may increase the number of photos \033[0m")
    flag = True
    while flag:
        key = input('\033[4;33mcontinue ?(y/n) :\033[0m')
        if key == 'y':
            flag = False
        if key == 'n':
            return


def backup():
    pass
