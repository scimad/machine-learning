import numpy as np
import matplotlib.pyplot as plt

class FunctionFactory:
    def __init__(self):
        pass

    def linear_function(*args):
        domain_points = args[0]
        x = domain_points
        a = args[1][0]
        b = args[1][1]

        return a*x+b

class DataFactory:
    def __init__(self):
        pass

    def create_sample_quadratic_dataset(self):
        return self.create_quadratic(-2, 3, 1, 1000, 10)

    @staticmethod
    def show_plot(x,y,xlabel='x', ylabel='y'):
        plt.scatter(x, y, s=0.5)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def create_linear(self, a, b, N=1000, noise_sigma=1):
        x = np.random.uniform(-1.25*b/a, 1.25*b/a, N)
        y = a*x+b + np.random.normal(scale=noise_sigma, size=x.shape)
        return x , y
    
    def create_quadratic(self, a, b, c, N=1000, noise_sigma=1):
        x = np.random.uniform(-1.25*b/2*a, 1.25*b/2*a, N)
        y = a*x*x+b*x+c + np.random.normal(scale=noise_sigma, size=x.shape)
        return (x, y)



class LeastSquareSolver:
    '''
    Description for class LeastSquareSolver

    attributes:
    observation_functions:  This is a set of function that computes the expected observation
                            or predicted observations from the current state x

    measurement:            Set of actual measurements / observation z_i of the state x

    state_vector:           Set of parameters representing the state x
    '''
    def __init__(self,system = 'Linear', function=None, domain_points = None, observations=None):
        self.state_vector = None
        self.domain_points = domain_points
        self.observations = observations
        self.predicted = None
        self.observation_functions = function
        self.squared_error = 9999999
        
        if system == 'Linear':
            self.state_vector = [0]*2
        elif system == 'Quadratic':
            self.state_vector = [0]*3

    def initialize_state_randomly(self):
        self.state_vector = np.random.random((len(self.state_vector)))

    def compute_predicted_observation(self):
        self.predicted = self.observation_functions(self.domain_points, self.state_vector)

    def error_fn(self):
        self.squared_error = np.linalg.norm(self.observations - self.predicted)
        return self.squared_error
    
    def add_to_visualizer(self,x,y,color='green'):
        plt.scatter(x, y, s=0.5, color=color)
    
    def visualize(self):
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    
    def improvise_state(self):
        pass

    def solve(self):
        self.initialize_state_randomly()
        while True:
            self.compute_predicted_observation()
            self.error_fn()
            self.improvise_state()
            self.add_to_visualizer(domain_points, self.predicted, color='red')
            self.add_to_visualizer(domain_points, observations, color='blue')
            self.visualize()

if __name__ == '__main__':
    df = DataFactory()
    
    domain_points, observations = df.create_linear(1, 4, 1000, noise_sigma=2)

    solver = LeastSquareSolver('Linear', FunctionFactory.linear_function, domain_points, observations)

    solver.solve()
