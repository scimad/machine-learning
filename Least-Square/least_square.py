import numpy as np
import matplotlib.pyplot as plt



class FunctionTemplate:
    def __init__(self):
        pass

    def func(self, *args):
        '''
        Should return predicted observations
        '''
        pass

    def get_sample_data(param_matrix, N, noise_sigma):
        pass

    def jacobian_of_error_func(self):
        pass



class LinearFunction(FunctionTemplate):
    def __init__(self):
        super().__init__()

    def func(self, domain_points, state_matrix):
        x = domain_points
        a, b = state_matrix[0][0], state_matrix[0][1]
        return a*x+b

    def initial_state_matrix(self):
        return np.random.normal(size=(1,2))

    def get_sample_data(self, param_matrix = [1, 1], N=1000, noise_sigma=[1]):
        a, b = param_matrix
        x = np.random.uniform(-1.25*b/a, 1.25*b/a, (N, 1, 1))
        y = a*x+b + np.random.normal(scale=noise_sigma[0], size=x.shape)
        return x , y



class ProjectionFunction(FunctionTemplate):
    def __init__(self):
        super().__init__()

    def func(self, domain_points, state_matrix):
        return [np.matmul(state_matrix, np.transpose(x)) for x in domain_points]
    
    def initial_state_matrix(self):
        return np.random.normal(size=(2,3))

    def get_sample_data(self, param_matrix=np.ones((2,3)), N=1000, noise_sigma=[1,2]):
        '''
        Function to simulate data from this:

        x1          a   b   c       x1
            =                       x2
        x2          d   e   f       x3

        '''
        X = np.random.uniform(-100, 100, (N, 1, 3))
        noise_data = np.concatenate((np.random.normal(scale=noise_sigma[0], size=(N, 1, 1)), np.random.normal(scale=noise_sigma[1], size=(N, 1, 1))), axis=1)
        Y = [np.matmul(param_matrix, np.transpose(x)) for x in X] + noise_data
        return X, Y



class LeastSquareSolver:
    '''
    Description for class LeastSquareSolver

    attributes:
    observation_func    :  This is a set of object that represents the observation function and what not!

    measurement:            Set of actual measurements / observation z_i of the state x

    state_matrix:           Set of parameters representing the state x
    '''
    def __init__(self, observation_func=None, domain_points = None, observations=None):
        self.state_matrix = None
        self.domain_points = domain_points
        self.observations = observations
        self.predicted = None
        self.observation_func = observation_func
        self.squared_error = [999999999]*len(observations)
        self.sum_of_squared_error = [999999999]
        # self.state_matrix = observation_func.initial_state_matrix()

    def initialize_state_randomly(self):
        self.state_matrix = observation_func.initial_state_matrix()

    def compute_predicted_observation(self):
        self.predicted = self.observation_func.func(self.domain_points, self.state_matrix)

    def compute_error(self):
        self.error_matrix = np.transpose(self.observations - self.predicted, axes=(0,2,1))
        self.squared_error = list(map(lambda err: np.dot(err[0], np.transpose(err[0])), self.error_matrix))
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
        pass

    def solve(self):
        self.initialize_state_randomly()
        iteration = 1 
        while True:
            self.compute_predicted_observation()
            self.compute_error()
            self.improvise_state()
            print (f"Sum of squared error in iteration {iteration} is {self.sum_of_squared_error}")
            input()
            # self.add_to_visualizer(domain_points, self.predicted, color='red')
            # self.add_to_visualizer(domain_points, observations, color='blue')
            # self.visualize()



if __name__ == '__main__':
    observation_func = LinearFunction()
    domain_points, observations = observation_func.get_sample_data([1,4], 1000, noise_sigma=[2])
    solver = LeastSquareSolver(observation_func, domain_points, observations)

    # observation_func = ProjectionFunction()
    # domain_points, observations = observation_func.get_sample_data(N=1000, noise_sigma=[2,3])
    # solver = LeastSquareSolver(observation_func, domain_points, observations)

    solver.solve()
