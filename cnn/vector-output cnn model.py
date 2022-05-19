import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D

'''多并行序列'''
def split_sequences(first_seq, secend_seq, sw_width):
    '''
    该函数将序列数据分割成样本
    '''
    input_seq1 = np.array(first_seq).reshape(len(first_seq), 1)
    input_seq2 = np.array(secend_seq).reshape(len(secend_seq), 1)
    out_seq = np.array([first_seq[i] + secend_seq[i] for i in range(len(first_seq))])
    out_seq = out_seq.reshape(len(out_seq), 1)

    dataset = np.hstack((input_seq1, input_seq2, out_seq))
    print('dataset:\n', dataset)

    X, y = [], []

    for i in range(len(dataset)):
        end_element_index = i + sw_width
        if end_element_index > len(dataset) - 1:
            break

        # 该语句实现步长为1的滑动窗口截取数据功能；
        seq_x, seq_y = dataset[i:end_element_index, :], dataset[end_element_index, :]

        X.append(seq_x)
        y.append(seq_y)

        process_X, process_y = np.array(X), np.array(y)

    n_features = process_X.shape[2]
    print('train_X:\n{}\ntrain_y:\n{}\n'.format(process_X, process_y))
    print('train_X.shape:{},trian_y.shape:{}\n'.format(process_X.shape, process_y.shape))
    print('n_features:', n_features)
    return process_X, process_y, n_features


def oned_cnn_model(sw_width, n_features, X, y, test_X, epoch_num, verbose_set):
    model = Sequential()

    model.add(Conv1D(filters=64, kernel_size=2, activation='relu',
                     strides=1, padding='valid', data_format='channels_last',
                     input_shape=(sw_width, n_features)))

    model.add(MaxPooling1D(pool_size=2, strides=None, padding='valid',
                           data_format='channels_last'))

    model.add(Flatten())

    model.add(Dense(units=50, activation='relu',
                    use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', ))

    model.add(Dense(units=n_features))

    model.compile(optimizer='adam', loss='mse',
                  metrics=['accuracy'], loss_weights=None, sample_weight_mode=None, weighted_metrics=None,
                  target_tensors=None)

    print('\n', model.summary())

    history = model.fit(X, y, batch_size=32, epochs=epoch_num, verbose=verbose_set)

    yhat = model.predict(test_X, verbose=0)
    print('\nyhat:', yhat)

    return model, history


if __name__ == '__main__':
    train_seq1 = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    train_seq2 = [15, 25, 35, 45, 55, 65, 75, 85, 95]
    sw_width = 3

    epoch_num = 3000
    verbose_set = 0

    train_X, train_y, n_features = split_sequences(train_seq1, train_seq2, sw_width)

    # 预测
    x_input = np.array([[70, 75, 145], [80, 85, 165], [90, 95, 185]])
    x_input = x_input.reshape((1, sw_width, n_features))

    model, history = oned_cnn_model(sw_width, n_features, train_X, train_y, x_input, epoch_num, verbose_set)

    print('\ntrain_acc:%s' % np.mean(history.history['accuracy']), '\ntrain_loss:%s' % np.mean(history.history['loss']))
