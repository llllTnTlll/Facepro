import face_encoding
import train_model
import face_recognition


def show_menu():
    # 展示菜单
    print('\033[1;32m=======================Function_Menu========================\033[0m')
    print('press 1 for : FaceRecognition(SVM)')
    print('press 2 for : LoadNewFace')
    print('press 3 for : FaceEncoding & TrainModel')
    print('press 4 for : Settings')
    print('press 5 for : Exit')
    key = input('\033[4;33menter num and press enter : \033[0m')
    function_choose(key)


def function_choose(num):
    # 功能选择
    numbers = {
        '1': do_recognition,
        '2': do_load,
        '3': do_train,
        '4': do_settings,
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


def do_load():
    # 向字典添加新的人脸
    print("do_load")


def do_train():
    # 对face_directory进行人脸编码
    flag = face_encoding.do_embedding()
    if flag == '':
        train_model.do_modeltrain()
    # 返回功能选择
    show_menu()


def main():
    show_menu()


if __name__ == '__main__':
    main()
