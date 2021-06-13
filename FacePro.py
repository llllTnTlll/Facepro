import face_encoding
import train_model
import face_recognition
import face_load
import cfg_manager
import user_management
import sys


def show_menu():
    # 展示菜单
    print('\033[1;32m=======================Function_Menu========================\033[0m')
    print('press 1 for : FaceRecognition(SVM)')
    print('press 2 for : FaceEncoding & TrainModel')
    print('press 3 for : LoadNewFace')
    print('press 4 for : UserManagement')
    print('press 5 for : Settings')
    print('press 6 for : Exit')
    key = input('\033[4;33menter num and press enter : \033[0m')
    function_choose(key)


def function_choose(num):
    # 功能选择
    numbers = {
        '1': do_recognition,
        '2': do_train,
        '3': do_load,
        '4': do_facemanage,
        '5': do_settings,
        '6': do_exit,
    }

    numbers.get(num, default)()


def default():
    # 在输入不属于字典的function_num时显示提示
    print('\033[1;31mError:no such function\033[0m')
    key = input('\033[4;33mpress num and press enter :\033[0m')
    function_choose(key)


def do_recognition():
    # 启动人脸识别
    face_recognition.do_recognition()
    main()


def do_load():
    face_load.startLoad()
    main()


def do_train():
    # 对face_directory进行人脸编码
    flag = face_encoding.do_embedding()
    if flag == '':
        train_model.do_modeltrain()
    # 返回功能选择
    main()


def do_settings():
    print('\033[1;32m=======================Setting_Menu========================\033[0m')
    print('press 1 for : ShowSettingTree')
    print('press 2 for : ChangeSettings')
    print('press 3 for : BackUp')
    key = input('\033[4;33minput num then press enter :\033[0m')
    if key == '1':
        cfg_manager.show_cfgtree()
    elif key == '2':
        cfg_manager.change_cfg()
    elif key == '3':
        main()
    do_settings()


def do_facemanage():
    user_management.do_management()
    main()


def do_exit():
    sys.exit()


def main():
    show_menu()


if __name__ == '__main__':
    main()
