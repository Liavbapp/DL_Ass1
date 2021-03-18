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
        A = np.random.rand(5, )
        W = np.random.rand(3, 5)
        b = np.random.rand(3, )
        expected_dim = (3, 1)
        Z_dim = forward.linear_forward(A, W, b)[0].shape
        self.assertTrue(expected_dim == Z_dim)

    def test_apply_batchnorm_t1(self):
        A = np.array([1, 2, 3, 4, 5, 6])
        expected_return = np.array([-1.4638501094227998, -0.8783100656536799, -0.29277002188455997,  0.29277002188455997,
                                    0.8783100656536799, 1.4638501094227998])
        actual_return = forward.apply_batchnorm(A)
        print(expected_return == actual_return)
        self.assertTrue(np.array_equal(expected_return, actual_return))

    def test_apply_batchnorm_t2(self):
        A = np.array([1])
        expected_return = np.array([0.])
        actual_return = forward.apply_batchnorm(A)
        print(expected_return == actual_return)
        self.assertTrue(np.array_equal(expected_return, actual_return))