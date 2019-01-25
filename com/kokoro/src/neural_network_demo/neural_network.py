import numpy
import scipy.special


class NeuralNetwork:

    def __init__(self, input_nodes, hide_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hide_nodes = hide_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.wih = numpy.random.normal(0.0, pow(self.input_nodes, -0.5), (self.hide_nodes, self.input_nodes))
        self.who = numpy.random.normal(0.0, pow(self.hide_nodes, -0.5), (self.output_nodes, self.hide_nodes))
        self.s_function = lambda x: \
            scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        # print('计算隐藏层输入, 结果为: ' + hidden_inputs)
        hidden_outputs = self.s_function(hidden_inputs)
        # print('计算隐藏层输出, 结果为: ' + hidden_outputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # print('计算输出层输入, 结果为: ' + final_inputs)
        final_outputs = self.s_function(final_inputs)
        # print('计算输出层输出, 结果为: ' + final_outputs)
        # 计算最终输出节点的误差(target - actual)
        output_errors = targets - final_outputs
        # print('计算输出层误差, 结果为: ' + final_outputs)
        # 计算隐藏节点的误差
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # 计算隐藏层和输出层之间新的权重
        self.who += self.learning_rate * numpy.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs)
        )
        # 计算输入层和隐藏层之间新的权重
        self.wih += self.learning_rate * numpy.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs)
        )

    def query(self, inputs_list):
        return self._calculate_final_outputs(inputs_list)

    def _calculate_final_outputs(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.s_function(hidden_inputs)
        final_input = numpy.dot(self.who, hidden_outputs)
        final_output = self.s_function(final_input)
        return final_output
