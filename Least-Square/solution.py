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

    def get_sample_data(self, state_matrix = [1, 1], N=1000, noise_sigma=[1]):
        a, b = state_matrix
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

    def get_sample_data(self, state_matrix = [1, 1, 1], N=1000, noise_sigma=[1]):
        a, b, c = state_matrix
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

    def get_sample_data(self, state_matrix = [1, 1, 1], N=1000, noise_sigma=[1]):
        a, b, c = state_matrix
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
        super().__init__(learning_rate=0.1)

    def func(self, domain_points, state_matrix):
        x = domain_points
        a, b = state_matrix[0][0], state_matrix[0][1]
        return a*np.exp(b*x)

    def initialize_state_matrix(self):
        return np.random.normal(size=(1,2))

    def get_sample_data(self, state_matrix = [1, 1, 1], N=1000, noise_sigma=[1]):
        a, b = state_matrix
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
        super().__init__(learning_rate=0.1)

    def func(self, domain_points, state_matrix):
        return [np.matmul(state_matrix, np.transpose(x)) for x in domain_points]

    def initialize_state_matrix(self):
        return np.random.normal(size=(2,3))

    def get_sample_data(self, state_matrix=np.ones((2,3)), N=1000, noise_sigma=[1,2]):
        '''
        Function to simulate data from this:

        x1          a   b   c       x1
            =                       x2
        x2          d   e   f       x3

        '''
        X = np.random.uniform(-100, 100, (N, 1, 3))
        noise_data = np.concatenate((np.random.normal(scale=noise_sigma[0], size=(N, 1, 1)), np.random.normal(scale=noise_sigma[1], size=(N, 1, 1))), axis=1)
        Y = [np.matmul(state_matrix, np.transpose(x)) for x in X] + noise_data
        return X, Y



class LeastSquareSolver:
    '''
    Description for class LeastSquareSolver

    attributes:
    observation_func    :  This is a set of object that represents the observation function and what not!

    observations:            Set of actual observations / observation z_i of the state x

    state_matrix:           Set of parameters representing the state x
    '''
    def __init__(self, observation_func=None, domain_points = None, observations=None):
        self.state_matrix = None
        self.domain_points = domain_points
        self.observations = observations
        self.predicted = None
        self.observation_func = observation_func
        self.squared_error = [999999999]*len(observations)
        self.sum_of_squared_error = 999999999
        self.previous_sum_of_squared_error = 999999999
        self.iterations_before_visualization = 1

    def initialize_state_randomly(self):
        self.state_matrix = observation_func.initialize_state_matrix()

    def compute_predicted_observation(self):
        self.predicted = self.observation_func.func(self.domain_points, self.state_matrix)

    def compute_error(self):
        self.e_i = self.observations - self.predicted
        self.squared_error = self.e_i * self.e_i # self.squared_error = np.array(list(map(lambda err: np.dot(np.transpose(err), err), self.e_i)))
        self.sum_of_squared_error = np.sum(self.squared_error)
        return self.sum_of_squared_error
    
    def add_to_visualizer(self,x,y,color='green'):
        plt.scatter(x, y, s=0.5, color=color)
    
    def visualize(self):
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    
    def improvise_state(self):
        # SGD or Gauss Netwon or Levenberg–Marquardt
        c_i = self.squared_error
        j_i = self.observation_func.jacobian_of_error_func(self.domain_points, self.observations, self.state_matrix)
        b_i = np.matmul(self.e_i, j_i)
        h_i = np.array(list(map(lambda j: np.dot(np.transpose(j), j), j_i)))

        c = np.sum(c_i,axis=0)
        b = np.transpose(np.sum(b_i, axis=(0,1)))
        H = np.sum(h_i, axis=0)
        learning_rate = self.observation_func.learning_rate
        change = - learning_rate * np.transpose(np.matmul(np.linalg.inv(H), b))

        self.state_matrix +=  change

    def solve(self):
        self.initialize_state_randomly()
        print ("The state_matrix is initialed as:", np.round(self.state_matrix,decimals=2))

        iterations = 0
        while True:
            iterations += 1
            self.compute_predicted_observation()
            self.compute_error()
            self.improvise_state()
            if iterations%self.iterations_before_visualization == 0:
                print (f'Iteration = {iterations}, state_matrix = {self.state_matrix}, L2 error= {self.sum_of_squared_error}')
                self.add_to_visualizer(domain_points, observations, color='green')
                self.add_to_visualizer(domain_points, self.predicted, color='red')
                self.visualize()



if __name__ == '__main__':
    # observation_func = LinearFunction()
    # domain_points, observations = observation_func.get_sample_data([1,-4], 1000, noise_sigma=[2])

    # observation_func = QuadraticFunction()
    # domain_points, observations = observation_func.get_sample_data([-2, 5, 2], N=1000, noise_sigma=[1.5])

    # observation_func = SineFunction()
    # domain_points, observations = observation_func.get_sample_data([1, 2, -3], N=1000, noise_sigma=[0.1])

    observation_func = ExpFunction()
    domain_points, observations = observation_func.get_sample_data([2, 2], N=1000, noise_sigma=[1])

    solver = LeastSquareSolver(observation_func, domain_points, observations)
    solver.solve()
