import numpy as np
import matplotlib.pyplot as plt

class FunctionTemplate:
    def __init__(self, learning_rate=1):
        self.learning_rate = learning_rate

    def func(self, *args):
        '''
        Should return predicted observations
        '''
        pass

    def visualize(self, x, y):
        plt.scatter(x,y,marker='x',s=1)
        plt.xlabel('x (data point)')
        plt.ylabel('y (observations)')
        plt.show()

    def get_sample_data(state_matrix, N, noise_sigma):
        pass

    def jacobian_of_error_func(self):
        pass



class LinearFunction(FunctionTemplate):
    def __init__(self):
        super().__init__(learning_rate=0.5)

    def func(self, domain_points, state_matrix):
        x = domain_points
        a, b = state_matrix[0][0], state_matrix[0][1]
        return a*x+b

    def initialize_state_matrix(self):
        return np.random.normal(size=(1,2))

    def get_sample_data(self, state_matrix = np.ones((1,2)), N=1000, noise_sigma=[1]):
        a, b = state_matrix[0][0], state_matrix[0][1]
        x = np.random.uniform(-1.25*b/a, 1.25*b/a, (N, 1, 1))
        y = a*x+b + np.random.normal(scale=noise_sigma[0], size=x.shape)
        self.visualize(x, y)
        return x, y

    def jacobian_of_error_func(self, domain_points, observations, state_matrix):
        a, b = state_matrix[0][0], state_matrix[0][1]
        x_i = np.array((domain_points))
        j_11 = -x_i
        j_12 = -np.ones(j_11.shape)

        j_i = np.concatenate((j_11, j_12), axis=2)
        return j_i

class QuadraticFunction(FunctionTemplate):
    def __init__(self):
        super().__init__(learning_rate=0.5)

    def func(self, domain_points, state_matrix):
        x = domain_points
        a, b, c = state_matrix[0][0], state_matrix[0][1], state_matrix[0][2]
        return a*(x*x) + b*x + c

    def initialize_state_matrix(self):
        return np.random.normal(size=(1,3))

    def get_sample_data(self, state_matrix = np.ones((1,3)), N=1000, noise_sigma=[1]):
        a, b, c = state_matrix[0][0], state_matrix[0][1], state_matrix[0][2]
        state_matrix = np.array(state_matrix)
        x = np.random.uniform(-1.25*b/a, 1.25*b/a, (N, 1, 1))
        y = a*x*x + b*x + c + np.random.normal(scale=noise_sigma[0], size=x.shape)
        # y = self.func(x, state_matrix)+ np.random.normal(scale=noise_sigma[0], size=x.shape)
        self.visualize(x, y)
        return x, y

    def jacobian_of_error_func(self, domain_points, observations, state_matrix):
        a, b,c = state_matrix[0][0], state_matrix[0][1], state_matrix[0][1]

        x_i = np.array((domain_points))

        j_11 = -x_i*x_i
        j_12 = -x_i
        j_13 = -np.ones(j_11.shape)

        j_i = np.concatenate((j_11, j_12, j_13), axis=2)
        return j_i



class SineFunction(FunctionTemplate):
    def __init__(self):
        super().__init__(learning_rate=0.25)

    def func(self, domain_points, state_matrix):
        x = domain_points
        a, b, c = state_matrix[0][0], state_matrix[0][1], state_matrix[0][2]
        return a*np.sin(b*x + c)

    def initialize_state_matrix(self):
        return np.random.normal(size=(1,3))

    def get_sample_data(self, state_matrix = np.ones((1,3)), N=1000, noise_sigma=[1]):
        a, b, c = state_matrix[0][0], state_matrix[0][1], state_matrix[0][2]
        state_matrix = np.array(state_matrix)
        x = np.random.uniform(-np.pi, np.pi, (N, 1, 1))
        y = a*np.sin(b*x + c) + np.random.normal(scale=noise_sigma[0], size=x.shape)
        # y = self.func(x, state_matrix)+ np.random.normal(scale=noise_sigma[0], size=x.shape)
        self.visualize(x, y)
        return x , y

    def jacobian_of_error_func(self, domain_points, observations, state_matrix):
        a, b,c = state_matrix[0][0], state_matrix[0][1], state_matrix[0][2]

        x_i = np.array((domain_points))

        j_11 = -np.sin(b*x_i + c)
        j_12 = -a*x_i*np.cos(b*x_i + c)
        j_13 = -a*np.cos(b*x_i + c)

        j_i = np.concatenate((j_11, j_12, j_13), axis=2)
        return j_i

class ExpFunction(FunctionTemplate):
    def __init__(self):
        super().__init__(learning_rate=0.2)

    def func(self, domain_points, state_matrix):
        x = domain_points
        a, b = state_matrix[0][0], state_matrix[0][1]
        return a*np.exp(b*x)

    def initialize_state_matrix(self):
        return np.random.normal(size=(1,2))

    def get_sample_data(self, state_matrix = np.ones((1,3)), N=1000, noise_sigma=[1]):
        a, b = state_matrix[0][0], state_matrix[0][1]
        state_matrix = np.array(state_matrix)
        x = np.random.uniform(-1, 1, (N, 1, 1))
        y = a*np.exp(b*x) + np.random.normal(scale=noise_sigma[0], size=x.shape)
        # y = self.func(x, state_matrix)+ np.random.normal(scale=noise_sigma[0], size=x.shape)
        self.visualize(x, y)
        return x , y

    def jacobian_of_error_func(self, domain_points, observations, state_matrix):
        a, b = state_matrix[0][0], state_matrix[0][1]

        x_i = np.array((domain_points))

        j_11 = -np.exp(b*x_i)
        j_12 = -a*x_i*np.exp(b*x_i)

        j_i = np.concatenate((j_11, j_12), axis=2)
        return j_i



class ProjectionFunction(FunctionTemplate):
    def __init__(self):
        super().__init__(learning_rate=0.5)

    def func(self, domain_points, state_matrix):
        return np.matmul(state_matrix, domain_points)

    def initialize_state_matrix(self):
        return np.random.normal(size=(2,3))
        # return np.random.normal(size=(2,3))

    def get_sample_data(self, state_matrix=np.ones((2,3)), N=1000, noise_sigma=[1,2]):
        '''
        Function to simulate data from this:

        x1          a   b   c       x1
            =                       x2
        x2          d   e   f       x3

        '''
        X = np.random.uniform(-100, 100, (N, 3, 1))
        Y = np.matmul(state_matrix, X)   + np.random.normal(size=(N, 2, 1))
        # Y = self.func(X, state_matrix) + np.random.normal(size=(N, 2, 1))
        return X, Y

    def jacobian_of_error_func(self, domain_points, observations, state_matrix):
        x_i = np.array((domain_points))
        j_i = - np.moveaxis(np.vstack((x_i.transpose(), x_i.transpose())), 2,0)
        return j_i
