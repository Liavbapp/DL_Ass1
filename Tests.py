import unittest
import forward
import numpy as np


class Tests(unittest.TestCase):

    def test_initialize_parameters_t1(self):
        dim = [3, 3, 2]
        expected_dim = {'W1': (3, 3), 'W2': (2, 3), 'b1': (3,), 'b2': (2,)}
        actual_dim = {key: val.shape for key, val in forward.initialize_parameters(dim).items()}
        self.assertTrue(expected_dim == actual_dim)

    def test_initialize_parameters_t2(self):
        dim = [10, 1]
        expected_dim = {'W1': (1, 10), 'b1': (1,)}
        actual_dim = {key: val.shape for key, val in forward.initialize_parameters(dim).items()}
        self.assertTrue(expected_dim == actual_dim)

    def test_initialize_parameters_t3(self):
        dim = [3, 8, 4, 100, 7, 5, 1]
        expected_dim = {'W1': (8, 3), 'W2': (4, 8), 'W3': (100, 4), 'W4': (7, 100), 'W5': (5, 7), 'W6': (1, 5),
                        'b1': (8,), 'b2': (4,), 'b3': (100,), 'b4': (7,), 'b5': (5,), 'b6': (1,)}
        actual_dim = {key: val.shape for key, val in forward.initialize_parameters(dim).items()}
        self.assertTrue(expected_dim == actual_dim)

    def test_initialize_parameters_t4(self):
        dim = [10]
        expected_dim = {}
        actual_dim = {key: val.shape for key, val in forward.initialize_parameters(dim).items()}
        self.assertTrue(expected_dim == actual_dim)

    def test_initialize_parameters_t5(self):
        dim = []
        expected_dim = {}
        actual_dim = {key: val.shape for key, val in forward.initialize_parameters(dim).items()}
        self.assertTrue(expected_dim == actual_dim)

    def test_linear_forward_t1(self):
        A_prev = np.random.rand(5, )
        W = np.random.rand(3, 5)
        b = np.random.rand(3, )
        expected_dim = (3,)
        Z_dim = forward.linear_forward(A_prev, W, b)[0].shape
        self.assertTrue(expected_dim == Z_dim)

    def test_linear_forward_t2(self):
        A_prev = np.array([0.91208099, 0.84926712, 0.34659675, 0.6648843, 0.00323112])
        W = np.array([[0.23212763, 0.17373304, 0.97972341, 0.90126425, 0.97650933],
                      [0.28495502, 0.61750353, 0.67666006, 0.46393482, 0.49559188],
                      [0.56688255, 0.64774301, 0.01890737, 0.2893646, 0.74474502]])
        b = np.array([4.13613966e-04, 7.75099399e-01, 6.88698967e-01])
        expected_Z = np.array([1.30163919, 2.10401938, 1.95720217])
        actual_Z = forward.linear_forward(A_prev, W, b)[0]
        np.testing.assert_allclose(actual_Z, expected_Z)

    def test_linear_activation_forward_t1(self):
        A_prev = np.random.rand(5, )
        W = np.random.rand(3, 5)
        b = np.random.rand(3, )
        activation = "relu"
        res_cur_active, res_dict = forward.linear_activation_forward(A_prev, W, b, activation)
        self.assertTrue(len(res_cur_active) == 3 and set(res_dict.keys()) == {'A', 'W', 'b', 'Z'})

    def test_linear_activation_forward_t2(self):
        A_prev = np.array([0.64333428, 0.394551, 0.19479036, 0.00881926, 0.41231911])
        W = np.array([[0.51855695, 0.30829058, 0.53081532, 0.38118337, 0.67695411],
                      [0.00768971, 0.05420387, 0.69827948, 0.27840483, 0.50860213],
                      [0.39872216, 0.39489627, 0.29277752, 0.10742955, 0.39920078]])
        b = np.array([0.26883461, 0.12902709, 0.19538913])
        activation = "relu"
        expected_A = np.array([1.10995701, 0.50354015, 0.83028329])
        expected_dict_keys = {'A', 'W', 'b', 'Z'}
        res_A, res_dict = forward.linear_activation_forward(A_prev, W, b, activation)
        # np.testing.assert_allclose(np.array_equal(res_A, expected_A))
        np.testing.assert_allclose(res_A, expected_A)
        self.assertTrue(set(res_dict.keys()) == expected_dict_keys)

    def test_linear_activation_forward_t3(self):
        A_prev = np.array([0.64333428, 0.394551, 0.19479036, 0.00881926, 0.41231911])
        W = np.array([[0.51855695, 0.30829058, 0.53081532, 0.38118337, 0.67695411],
                      [0.00768971, 0.05420387, 0.69827948, 0.27840483, 0.50860213],
                      [0.39872216, 0.39489627, 0.29277752, 0.10742955, 0.39920078]])
        b = np.array([0.26883461, 0.12902709, 0.19538913])
        activation = "softmax"
        expected_A = np.array([0.43453103, 0.23695032, 0.32851865])
        expected_dict_keys = {'A', 'W', 'b', 'Z'}
        res_A, res_dict = forward.linear_activation_forward(A_prev, W, b, activation)
        # np.testing.assert_allclose(np.array_equal(res_A, expected_A))
        np.testing.assert_allclose(res_A, expected_A)
        self.assertTrue(set(res_dict.keys()) == expected_dict_keys)

    def test_apply_batchnorm_t1(self):
        A = np.array([1, 2, 3, 4, 5, 6])
        expected_return = np.array([-1.4638501094227998, -0.8783100656536799, -0.29277002188455997, 0.29277002188455997,
                                    0.8783100656536799, 1.4638501094227998])
        actual_return = forward.apply_batchnorm(A)
        np.testing.assert_allclose(expected_return, actual_return)

    def test_apply_batchnorm_t2(self):
        A = np.array([1])
        expected_return = np.array([0.])
        actual_return = forward.apply_batchnorm(A)
        np.testing.assert_allclose(expected_return, actual_return)

    def test_relu_t1(self):
        Z = np.array([1, 2, 4, 5])
        expected_res = np.array([1, 2, 4, 5])
        actual_res = forward.relu(Z)[0]
        self.assertTrue(np.array_equal(expected_res, actual_res))

    def test_relu_t2(self):
        Z = np.array([1, 2, 4, -3])
        expected_res = np.array([1, 2, 4, 0])
        actual_res = forward.relu(Z)[0]
        self.assertTrue(np.array_equal(expected_res, actual_res))

    def test_compute_cost(self):
        AL = np.array([[0.5, 0.3, 0.5, 0.4],
                       [0, 0.4, 0.1, 0.1],
                       [0.5, 0.3, 0.4, 0.4]])
        Y = np.array([[1, 0, 0, 0],
                      [0, 0, 0, 1],
                      [0, 1, 1, 0]])
        res_cost = forward.compute_cost(AL, Y)
        predicted_cost = 0
        self.assertTrue(res_cost == predicted_cost)
