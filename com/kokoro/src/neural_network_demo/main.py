import numpy
import os
from com.kokoro.src.neural_network_demo.neural_network import NeuralNetwork

PATH = os.path.abspath('')


def main():
    # 创建神经网络
    input_nodes = 784
    hide_nodes = 100
    output_nodes = 10
    learning_rate = 0.3
    network = NeuralNetwork(input_nodes, hide_nodes, output_nodes, learning_rate)
    # 准备数据
    data_file = open("resource/mnist_train_100.csv", 'r')
    data_list = data_file.readlines()
    data_file.close()
    print(len(data_list))
    image_data_list = []
    progress_total = len(data_list)
    progress_current = 0
    for data in data_list:
        progress_current += 1
        all_values = data.split(',')
        image_value = all_values[0]
        image_array = numpy.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01
        image_data_list.append(ImageData(image_value, image_array))
        print("数据转换中: " + str(progress_current) + "/" + str(progress_total))
    progress_total = len(image_data_list)
    progress_current = 0
    for train_data in image_data_list:
        progress_current += 1
        image_value = train_data.image_value
        image_array = train_data.image_array
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(image_value)] = 0.99
        network.train(image_array, targets)
        print("训练中: " + str(progress_current) + "/" + str(progress_total))
    print('训练结束, 开始测试')
    # train_data = image_data_list[3]
    # result = network.query(train_data.image_array)
    # result_num = transport_result(result)
    # print("图片值为: " + train_data.image_value + "识别结果为: " + str(result_num))
    total = len(image_data_list)
    right_cnt = 0
    error_train_data = []
    for train_data in image_data_list:
        target = train_data.image_value
        output = transport_result(network.query(train_data.image_array))
        report = "图片值为: " + train_data.image_value + "识别结果为: " + str(output)
        print(report)
        if target == str(output):
            right_cnt += 1
        else:
            train_data.report = report
            error_train_data.append(train_data)
    print("识别率为" + str(right_cnt / total) + ", 以下为识别错误的数字")
    # for train_data in error_train_data:
    #     print(train_data.report)


class ImageData:
    def __init__(self, image_value, image_array):
        self.image_value = image_value
        self.image_array = image_array


def transport_result(result):
    max_point = 0
    for i in range(0, len(result) - 1):
        if result[max_point] < result[i]:
            max_point = i
    return max_point


if __name__ == '__main__':
    main()
