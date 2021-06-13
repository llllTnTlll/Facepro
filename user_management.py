import pickle_helper
import cfg_manager
import os

def do_management():
    data = pickle_helper.load_pickle_from_disk(
        r'C:\Users\ZHIYUAN\PycharmProjects\Facepro\data\pickleHere\embeddings.pickle')
    names = data["names"]
    name_list = []
    for name in names:
        if name not in name_list:
            name_list.append(name)
    print('|  id  |      name      |')
    for i in range(len(name_list)):
        print(' %i       %s' % (i, name_list[i]))
    flag = True
    selected = ''
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