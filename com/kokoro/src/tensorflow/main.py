import tensorflow as tf
from tensorflow import keras
import com.kokoro.src.mnist.digital_mnist as mnist


def main():
    (train_labels, train_images) = mnist.load_data('resources/mnist_train_60000.csv')
    (test_labels, test_images) = mnist.load_data('resources/mnist_test_10000.csv')
    # 设置模型层结构, 输入层为28*28的灰度值, 隐藏层128节点, 输出层10节点
    model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    # 设置模型编译参数
    model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)
    print('结束')


if __name__ == '__main__':
    main()
