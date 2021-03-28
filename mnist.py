import tensorflow as tf
import numpy as np

# def load_data():
import model_trainer


def expand_y(y_data, num_classes=10):
    flatten_y = y_data.flatten()
    expanded_y = np.zeros((flatten_y.shape[0], num_classes))
    expanded_y[np.arange(flatten_y.shape[0]), flatten_y] = 1
    return expanded_y


def generate_validation_data(data_x, data_y, validation_size=0.2):
    combined_data = np.concatenate([data_x, data_y], axis=1)
    np.random.shuffle(combined_data)
    num_rows = int(validation_size * combined_data.shape[0])
    validation, train = combined_data[:num_rows, :], combined_data[num_rows:, :]
    train_x, train_y = train[:, :-1], train[:, -1:]
    validation_x, validation_y = validation[:, :-1], validation[:, -1:]
    return train_x, train_y, validation_x, validation_y


def reshape_data(data):
    data_x, data_y = data
    data_x = data_x.reshape(data_x.shape[0], data_x.shape[1] * data_x.shape[2])
    data_y = np.expand_dims(data_y.T, axis=1)
    return data_x, data_y


def pre_process():
    train_data, test_data = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    train_x, train_y = reshape_data(train_data)
    test_x, test_y = reshape_data(test_data)
    train_x, train_y, validation_x, validation_y = generate_validation_data(train_x, train_y)
    train_y, test_y, validation_y = expand_y(train_y), expand_y(test_y), expand_y(validation_y)

    # train_x, validation_x, test_x = normalize_input(train_x, validation_x, test_x)

    return {'train_x': train_x.T, 'train_y': train_y.T, 'test_x': test_x.T, 'test_y': test_y.T,
            'validation_x': validation_x.T, 'validation_y': validation_y.T}


def normalize_input(train_x, validation_x, test_x):
    conct_arr = np.concatenate([train_x, validation_x, test_x], axis=0)
    train_len, validation_len, test_len = train_x.shape[0], validation_x.shape[0], test_x.shape[0]
    avg = np.sum(conct_arr) / (conct_arr.shape[0] * conct_arr.shape[1])
    std = np.std(conct_arr)
    normalized = (conct_arr - avg) / (std ** 2)
    train_x = normalized[:train_len, :]
    validation_x = normalized[train_len:train_len+validation_len, :]
    test_x = normalized[train_len + validation_len:, :]
    return train_x, validation_x, test_x

def normalize_arr(arr):
    avg = np.sum(arr) / (arr.shape[0] * arr.shape[1])
    std = np.std(arr)
    return (arr - avg) / std ** 2


def normalize_data(data_dict):
    data_dict.update({'train_x': normalize_arr(data_dict['train_x']),
                      'test_x': normalize_arr(data_dict['test_x'])})
    return data_dict


def run_config():
    data_set = pre_process()
    # data_set = normalize_data(data_set)
    layers_dim = [784, 20, 7, 5, 10]
    lr = 0.009
    epochs = 1000
    batch_size = 1024
    params, costs = model_trainer.L_layer_model(X=data_set['train_x'], Y=data_set['train_y'], layers_dims=layers_dim,
                                                learning_rate=lr, num_iterations=epochs, batch_size=batch_size)
    print(model_trainer.predict(X=data_set['train_x'], Y=data_set['train_y'], parameters=params))
    # print(costs)


if __name__ == '__main__':
    run_config()
