import face_embedding
import train_model
import face_recognition


def show_menu():
    # 展示菜单
    print('\033[1;32m=======================Function_Menu========================\033[0m')
    print('press 1 for : FaceRecognition(SVM)')
    print('press 2 for : LoadNewFace')
    print('press 3 for : Settings')
    print('press 4 for : Exit')
    key = input('\033[4;33menter num and press enter : \033[0m')
    function_choose(key)


def function_choose(num):
    # 功能选择
    numbers = {
        '1': do_recognition,

    }

    numbers.get(num, default)()


def default():
    # 在输入不属于字典的function_num时显示提示
    print('\033[1;31mError:no such function\033[0m')
    key = input('\033[4;33mpress num and press enter :\033[0m')
    function_choose(key)


def do_recognition():
    flag = face_embedding.do_embedding()
    if flag == '':
        train_model.do_modeltrain()
        face_recognition.do_recognition()


def main():
    show_menu()


if __name__ == '__main__':
    main()
