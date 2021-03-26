import random
import unittest

import backward
import forward
import numpy as np


class TestBackward(unittest.TestCase):

    def test_softmax_backward_t1(self):
        A_L = np.random.rand(2, 3)
        T_L = np.array([[1, 0, 1],
                        [0, 1, 0]])
        dA = np.random.rand(2, 3)
        activation_cache = {'A_L': A_L, 'T_L': T_L}
        dZ = backward.softmax_backward(dA, activation_cache)
        self.assertTrue(dZ.shape == (2, 3))

    def test_softmax_backward_t2(self):
        A_L = np.array([[0.71453991, 0.66455975, 0.6709691],
                        [0.75502395, 0.124922, 0.0980855]])
        T_L = np.array([[1, 0, 1],
                        [0, 1, 0]])
        dA = np.array([[0.0584136, 0.47822658, 0.14694834],
                       [0.31270005, 0.62306705, 0.87186516]])
        activation_cache = {'A_L': A_L, 'T_L': T_L}
        dZ = backward.softmax_backward(dA, activation_cache)
        dZ_expected = np.array([[-0.01667475, 0.31781014, -0.04835054],
                                [0.23609603, -0.54523227, 0.08551733]])
        np.testing.assert_allclose(dZ, dZ_expected)

    def test_linear_backward_t1(self):
        dZ = np.random.rand(2,
                            5)  # dim: 2x5 (number of neurons current layer x number of instances in the batch). each row is gradient vector
        W = np.random.rand(2, 3)  # dim: 2x3 (cur_layer_nbr_neurons x prev_layer_nbr_neurons)
        b = np.random.rand(2)  # dim: 2 (number of neurons cur layer)
        A_prev = np.random.rand(3, 5)  # dim: 3x5 (number of neurons prev layer x number of instances in the batch)
        cache = {'A_prev': A_prev, 'W': W, 'b': b}
        dA_prev, dW, db = backward.Linear_backward(dZ, cache)
        self.assertTrue(dA_prev.shape == A_prev.shape)
        self.assertTrue(dW.shape == W.shape)
        self.assertTrue(db.shape == b.shape)

    def test_linear_backward_t2(self):
        dZ = np.array([[0.38264578, 0.82389831, 0.42980474, 0.24462336, 0.11402943],
                       [0.44298328, 0.69824283, 0.50412281, 0.94706762,
                        0.88424902]])  # dim: 2x5 (number of neurons current layer x number of instances in the batch). each row is gradient vector
        W = np.array([[0.02252559, 0.16737019, 0.17820524], [0.21318285, 0.88147624,
                                                             0.01027424]])  # dim: 2x3 (cur_layer_nbr_neurons x prev_layer_nbr_neurons)
        b = np.array([0.24736851, 0.47307659])  # dim: 2 (number of neurons cur layer)
        A_prev = np.array([[0.83753335, 0.20400821, 0.47304129, 0.37658064, 0.01167356],
                           [0.85881864, 0.70107786, 0.24634004, 0.97749331, 0.82313173],
                           [0.77861885, 0.05464926, 0.46090311, 0.01173639,
                            0.30267204]])  # dim: 3x5 (number of neurons prev layer x number of instances in the batch)
        cache = {'A_prev': A_prev, 'W': W, 'b': b}
        dA_prev, dW, db = backward.Linear_backward(dZ, cache)
        expected_dA_prev = np.array([[0.02061115, 0.03348244, 0.02343039, 0.04148177, 0.03821506],
                                     [0.09090455, 0.1506761, 0.10326176, 0.17515205, 0.15970593],
                                     [0.01454816, 0.03079938, 0.01635459, 0.01066471, 0.00588113]])
        expected_dW = np.array([[0.15706551, 0.26901945, 0.1156887],
                                [0.22378022, 0.52955124, 0.17883559]])
        expected_db = np.array([0.39900032, 0.69533311])
        np.testing.assert_allclose(dA_prev, expected_dA_prev, rtol=1e-5)
        np.testing.assert_allclose(dW, expected_dW, rtol=1e-5)
        np.testing.assert_allclose(db, expected_db, rtol=1e-5)


class TestsForward(unittest.TestCase):

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
        A_prev = np.random.rand(3, 7)
        W = np.random.rand(5, 3)
        b = np.random.rand(5, )
        expected_dim = (5, 7)
        Z_dim = forward.linear_forward(A_prev, W, b)[0].shape
        self.assertTrue(expected_dim == Z_dim)

    def test_linear_forward_t2(self):
        A_prev = np.array([[0.91208099, 0.84926712, 0.52362],
                           [0.33341, 0.45644, 0.11124]])  # dim: n=2 x m=3
        W = np.array([[0.23212763, 0.17373304],
                      [0.28495502, 0.61750353],
                      [0.56688255, 0.64774301]])  # dim: l=3 x l-1=2
        b = np.array([4.13613966e-04, 7.75099399e-01, 6.88698967e-01])
        expected_Z = np.array([[0.27005715, 0.27685069, 0.14128635],
                               [1.24088331, 1.29895564, 0.99299864],
                               [1.42170576, 1.4657895, 1.05758494]])
        actual_Z = forward.linear_forward(A_prev, W, b)[0]
        np.testing.assert_allclose(actual_Z, expected_Z)
        self.assertTrue(actual_Z.shape == (3, 3))

    def test_linear_activation_forward_t1(self):
        A_prev = np.random.rand(5, 8)  # dim: n=5 x m=8
        W = np.random.rand(7, 5)  # dim: l=7 x l-1=5
        b = np.random.rand(7, )
        activation = "relu"
        res_cur_active, res_dict = forward.linear_activation_forward(A_prev, W, b, activation)
        self.assertTrue(res_cur_active.shape == (7, 8) and set(res_dict.keys()) == {'A', 'W', 'b', 'Z'})

    def test_linear_activation_forward_t2(self):
        A_prev = np.array([[0.25821009, 0.82192552, 0.69363923, 0.1253511, 0.79755345,
                            0.08553971, 0.17986499, 0.96764057],
                           [0.17007695, 0.88872916, 0.44953625, 0.02003515, 0.82340014,
                            0.39599977, 0.49473922, 0.91576224],
                           [0.22212011, 0.35910357, 0.23401578, 0.59454932, 0.93764331,
                            0.59039333, 0.74805269, 0.74383216],
                           [0.49291735, 0.00226226, 0.5712783, 0.82571755, 0.67586954,
                            0.22033754, 0.6322879, 0.73067165],
                           [0.63116222, 0.50653914, 0.96360902, 0.44024258, 0.26422106,
                            0.35997056, 0.79055798, 0.60226783]])

        W = np.array([[0.4780447, 0.4316493, 0.9132166, 0.95332259, 0.29922231],
                      [0.19871669, 0.67676202, 0.03058338, 0.91666213, 0.54910392],
                      [0.13748834, 0.74540072, 0.35554872, 0.04446443, 0.72546013],
                      [0.32247701, 0.10526012, 0.84228229, 0.54938547, 0.85725642],
                      [0.62944918, 0.17553011, 0.01115437, 0.58370177, 0.28079927],
                      [0.63808557, 0.70744128, 0.51337424, 0.85654115, 0.96232575],
                      [0.5100887, 0.10277573, 0.05485349, 0.90112344, 0.15978727]])

        b = np.array([0.13685905, 0.33084046, 0.08170056, 0.76777732, 0.53512246,
                      0.85362075, 0.46276118])
        activation = "relu"
        expected_A = np.array([[1.19531944, 1.39505932, 1.70914452, 1.66728853, 2.45319931,
                                1.20560484, 1.95885743, 2.55078025],
                               [1.30245824, 1.38682789, 1.83285489, 1.38613512, 1.83987873,
                                1.03352879, 1.73797419, 2.16611547],
                               [0.80275176, 1.35241845, 1.31981801, 0.68135384, 1.3602298,
                                0.86949486, 1.34281067, 1.63122846],
                               [1.8679038, 1.86432051, 2.37579763, 2.14212544, 2.49921964,
                                1.76396006, 2.53300818, 2.72044741],
                               [1.19493044, 1.35604341, 1.65728741, 1.22976579, 1.66083149,
                                0.89475159, 1.33457968, 1.90885455],
                               [2.28231873, 2.68055115, 3.15100924, 2.38392331, 3.25957407,
                                2.02657882, 3.00477483, 3.70619868],
                               [1.16916614, 1.07603106, 1.54438143, 1.3757922, 1.65690379,
                                0.83554836, 1.34247931, 1.84592371]])
        expected_dict_keys = {'A', 'W', 'b', 'Z'}
        res_A, res_dict = forward.linear_activation_forward(A_prev, W, b, activation)
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
        Z = np.array([[1, 2, 4, 5]])
        expected_res = np.array([[1, 2, 4, 5]])
        actual_res = forward.relu(Z)[0]
        self.assertTrue(np.array_equal(expected_res, actual_res))

    def test_relu_t2(self):
        Z = np.array([[1, 2, 4, -3],
                      [-1, 5, 8, 0]])
        expected_res = np.array([[1, 2, 4, 0],
                                 [0, 5, 8, 0]])
        actual_res = forward.relu(Z)[0]
        self.assertTrue(np.array_equal(expected_res, actual_res))

    def test_softmax_t1(self):
        Z = np.array([[1, 2, 4, -3],
                      [-1, 5, 8, 0]])

        actual_res = forward.softmax(Z)[0]
        self.assertTrue(Z.shape == actual_res.shape)

    def test_softmax_t2(self):
        Z = np.array([[1, 2, 4, -3],
                      [-1, 5, 8, 0]])
        expected_res = np.array([[8.50660828e-04, 2.31233587e-03, 1.70859795e-02, 1.55803965e-05],
                                 [1.15124424e-04, 4.64445075e-02, 9.32862871e-01, 3.12940630e-04]])
        actual_res = forward.softmax(Z)[0]
        np.testing.assert_allclose(expected_res, actual_res)

    def test_compute_cost(self):
        AL = np.array([[0.5, 0.3, 0.5, 0.4],
                       [0.7, 0.4, 0.1, 0.1],
                       [0.1, 0.3, 0.4, 0.4]])

        Y = np.array([[1, 1, 0, 1],
                      [0, 1, 0, 1],
                      [1, 1, 1, 1]])
        expected_cost = 2.8428564756741324
        predicted_cost = forward.compute_cost(AL, Y)
        np.testing.assert_allclose(expected_cost, predicted_cost)

    def test_L_model_forward_t1(self):
        X = np.random.randn(3, 6)
        params = forward.initialize_parameters([3, 5, 7, 8, 2])
        use_batchnorm = False
        forward_result = forward.L_model_forward(X, params, use_batchnorm)[0]
        self.assertTrue(forward_result.shape == (2, 6))
