import os

def getCurrentPath():
    return os.path.dirname(os.path.realpath(__file__))


def getProjectPath():
    cur_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(os.path.dirname(cur_path))

