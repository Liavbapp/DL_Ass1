import backward
import forward
import numpy as np


def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size):
    """
    Implements a L-layer neural network.
    All layers but the last  have the ReLU activation function,
    and the final layer apply the softmax activation function
    :param X: the input data, a numpy array of shape (height*width , number_of_examples)
    :param Y: the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    :param layers_dims: a list containing the dimensions of each layer, including the input
    :param learning_rate: the learning rate used to update the parameters (the “alpha”)
    :param num_iterations:
    :param batch_size: the number of examples in a single training batch
    :return parameters: the parameters learnt by the system during the training
                        (the same parameters that were updated in the update_parameters function).
    :return costs: the values of the cost function.
                    One value is to be saved after each 100 training iterations (e.g. 3000 iterations -> 30 values).

    """
    combined_data = np.concatenate([X, Y], axis=0)
    m = X.shape[1]
    num_batches = int(m / batch_size)
    batches = np.array_split(combined_data, indices_or_sections=num_batches, axis=1)

    params = forward.initialize_parameters(layers_dims)
    costs = []
    for epoch in range(0, num_iterations):
        for batch in batches:
            X_batch = batch[0:X.shape[0], :]
            Y_batch = batch[X.shape[0]:, :]
            prediction, caches = forward.L_model_forward(X_batch, params, use_batchnorm=False)
            if epoch % 100 == 0:
                cost = forward.compute_cost(prediction, Y_batch)
                costs.append(cost)
                print(cost)
            grads = backward.L_model_backward(prediction, Y_batch, caches)
            params = backward.update_parameters(params, grads, learning_rate)

    return params, costs


def predict(X, Y, parameters):
    """
    The function receives an input data and the true labels and
    calculates the accuracy of the trained neural network on the data.

    :param X: the input data, a numpy array of shape (height*width, number_of_examples)
    :param Y: the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    :param parameters: a python dictionary containing the DNN architecture’s parameters
    :return accuracy: the accuracy measure of the neural net on the provided data
                    (i.e. the percentage of the samples for which the correct
                     label receives the hugest confidence score).
                     Using the softmax function to normalize the output values
    """
    # TODO: check what "use softmax function to normalize output values" means
    m = X.shape[1]
    prediction, caches = forward.L_model_forward(X, parameters, use_batchnorm=False)
    prediction_arg_max = np.argmax(prediction, axis=0)
    label_arg_max = np.argmax(Y, axis=0)
    correct_predictions = np.sum(prediction_arg_max == label_arg_max)
    accuracy = correct_predictions / m
    return accuracy

