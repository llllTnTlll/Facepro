import os


# 删除指定文件夹中的所有文件
def deleteAll(path: str, removedir: bool):
    """
    :param path: 清理目录
    :param removedir: 是否保留目录文件夹
    """
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    if removedir:
        os.removedirs(path)


# 获取指定目录的指定格式文件列表
def dirwalker(path: str, formatlist: tuple):
    """
    :param path: 获取路径
    :param formatlist: 目标格式元组
    :return: 对应文件路径列表
    """
    pathlist = []
    for parent, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.lower().endswith(formatlist):
                pathlist.append(os.path.join(parent, filename))
    return pathlist
