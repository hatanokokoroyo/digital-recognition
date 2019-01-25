import sys
import numpy as np


# 传入数字类型的mnist文件路径
def load_data(file_path, size=10000):
    """读取文件, 拆分数据, -> (data, label)"""
    data_list = []
    print('尝试读取文件')
    try:
        file = open(file_path, 'r')
    except FileNotFoundError as e:
        print('文件不存在, 请检查路径')
        return [], []
    data_count = 0
    for lines in _count_lines(file, size):
        data_list.extend(lines)
        data_count += len(lines)
        sys.stdout.write('计算数据容量中: {0}\r'.format(data_count))
        sys.stdout.flush()
    if data_count <= 0:
        print('数据容量为0')
        return [], []
    file.close()
    sys.stdout.write('计算数据容量中: {0}, 计算结束\r\n'.format(data_count))
    sys.stdout.flush()
    # 开始转化数据
    labels = []
    imgs = []
    cnt = 0
    total = len(data_list)
    for line in data_list:
        cnt += 1
        label, img = _convert_data(line)
        labels.append(label)
        imgs.append(img)
        sys.stdout.write('转换数据中: {0}/{1}\r'.format(cnt, total))
        sys.stdout.flush()
    sys.stdout.write('转换数据中: {0}/{1}, 转换结束\r\n'.format(cnt, total))
    sys.stdout.flush()
    return np.asarray(labels), np.asarray(imgs)


def _convert_data(line):
    """转换数据, 把原数据拆分为一个数字和一个数组 -> label, img"""
    data = line.split(',')
    data = list(map(int, data))
    label = data[0]
    img = data[1:]
    img = (np.asarray(img) / 255.0 * 0.99 + 0.01).reshape(28, 28)
    return label, img


def _count_lines(file, size):
    while 1:
        lines = file.readlines(size)
        if not lines:
            break
        yield lines
