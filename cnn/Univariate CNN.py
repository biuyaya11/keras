import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D

'''尽管传统上是针对二维图像数据开发的，但CNNs可以用来对单变量时间序列预测问题进行建模。
单变量时间序列是由具有时间顺序的单个观测序列组成的数据集，需要一个模型从过去的观测序列中学习以预测序列中的下一个值。'''
'''CNN实际不认为数据有时间同步，将其视为可执行卷积读取操作的序列'''

# 该该函数将序列数据分割成样本
def split_sequence(sequence, sw_width, n_features):
    '''
    这个简单的示例，通过for循环实现有重叠截取数据，滑动步长为1，滑动窗口宽度为sw_width。
    以后的文章，会介绍使用yield方法来实现特定滑动步长的滑动窗口的实例。
    '''
    X, y = [], []

    for i in range(len(sequence)):
        # 获取单个样本中最后一个元素的索引，因为python切片前闭后开，索引从0开始，所以不需要-1
        end_element_index = i + sw_width
        # 如果样本最后一个元素的索引超过了序列索引的最大长度，说明不满足样本元素个数，则这个样本丢弃
        if end_element_index > len(sequence) - 1:
            break
        # 通过切片实现步长为1的滑动窗口截取数据组成样本的效果
        seq_x, seq_y = sequence[i:end_element_index], sequence[end_element_index]

        X.append(seq_x)
        y.append(seq_y)

        process_X, process_y = np.array(X), np.array(y)
        process_X = process_X.reshape((process_X.shape[0], process_X.shape[1], n_features))

    print('split_sequence:\nX:\n{}\ny:\n{}\n'.format(np.array(X), np.array(y)))
    print('X_shape:{},y_shape:{}\n'.format(np.array(X).shape, np.array(y).shape))
    print('train_X:\n{}\ntrain_y:\n{}\n'.format(process_X, process_y))
    print('train_X.shape:{},trian_y.shape:{}\n'.format(process_X.shape, process_y.shape))
    return process_X, process_y


def oned_cnn_model(sw_width, n_features, X, y, test_X, epoch_num, verbose_set):
    model = Sequential()

    # 对于一维卷积来说，data_format='channels_last'是默认配置，该API的规则如下：
    # 输入形状为：(batch, steps, channels)；输出形状为：(batch, new_steps, filters)，padding和strides的变化会导致new_steps变化
    # 如果设置为data_format = 'channels_first'，则要求输入形状为： (batch, channels, steps).
    model.add(Conv1D(filters=64, kernel_size=2, activation='relu',
                     strides=1, padding='valid', data_format='channels_last',
                     input_shape=(sw_width, n_features)))

    # 对于一维池化层来说，data_format='channels_last'是默认配置，该API的规则如下：
    # 3D 张量的输入形状为: (batch_size, steps, features)；输出3D张量的形状为：(batch_size, downsampled_steps, features)
    # 如果设置为data_format = 'channels_first'，则要求输入形状为：(batch_size, features, steps)
    model.add(MaxPooling1D(pool_size=2, strides=None, padding='valid',
                           data_format='channels_last'))

    # data_format参数的作用是在将模型从一种数据格式切换到另一种数据格式时保留权重顺序。默认为channels_last。
    # 如果设置为channels_last，那么数据输入形状应为：（batch，…，channels）；如果设置为channels_first，那么数据输入形状应该为（batch，channels，…）
    # 输出为（batch, 之后参数尺寸的乘积）
    model.add(Flatten())

    # Dense执行以下操作：output=activation（dot（input，kernel）+bias），
    # 其中,activation是激活函数，kernel是由层创建的权重矩阵，bias是由层创建的偏移向量（仅当use_bias为True时适用）。
    # 2D 输入：(batch_size, input_dim)；对应 2D 输出：(batch_size, units)
    model.add(Dense(units=50, activation='relu',
                    use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', ))

    # 因为要预测下一个时间步的值，因此units设置为1
    model.add(Dense(units=1))

    # 配置模型
    model.compile(optimizer='adam', loss='mse',
                  metrics=['accuracy'], loss_weights=None, sample_weight_mode=None, weighted_metrics=None,
                  target_tensors=None)

    print('\n', model.summary())
    # X为输入数据，y为数据标签；batch_size：每次梯度更新的样本数，默认为32。
    # verbose: 0,1,2. 0=训练过程无输出，1=显示训练过程进度条，2=每训练一个epoch打印一次信息

    history = model.fit(X, y, batch_size=32, epochs=epoch_num, verbose=verbose_set)

    yhat = model.predict(test_X, verbose=0)
    print('\nyhat:', yhat)

    return model, history


if __name__ == '__main__':
    train_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    sw_width = 3
    n_features = 1
    epoch_num = 1000
    verbose_set = 0

    train_X, train_y = split_sequence(train_seq, sw_width, n_features)

    # 预测
    x_input = np.array([70, 80, 90])
    x_input = x_input.reshape((1, sw_width, n_features))

    model, history = oned_cnn_model(sw_width, n_features, train_X, train_y, x_input, epoch_num, verbose_set)

    print('\ntrain_acc:%s' % np.mean(history.history['accuracy']), '\ntrain_loss:%s' % np.mean(history.history['loss']))
