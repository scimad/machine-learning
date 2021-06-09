import numpy as np
import matplotlib.pyplot as plt

class FunctionFactory:
    def __init__(self):
        pass

    def linear_function(*args):
        measurements = args[0]
        x = measurements[0]
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
        return (x ,y)
    
    def create_quadratic(self, a, b, c, N=1000, noise_sigma=1):
        x = np.random.uniform(-1.25*b/2*a, 1.25*b/2*a, N)
        y = a*x*x+b*x+c + np.random.normal(scale=noise_sigma, size=x.shape)
        return (x, y)



class LeastSquareSolver:
    def __init__(self,system = 'Linear', function=None, measurements=None):
        self.state_vector = None
        self.measurement = measurements
        self.predicted = None
        self.function = function
        self.squared_error = 9999999
        
        if system == 'Linear':
            self.state_vector = [0]*2
        elif system == 'Quadratic':
            self.state_vector = [0]*3

    def initialize_state_randomly(self):
        self.state_vector = np.random.random((len(self.state_vector)))

    def compute_prediction(self):
        self.predicted = self.function(self.measurement, self.state_vector)

    def error_fn(self):
        self.squared_error = np.linalg.norm(self.measurement[1]-self.predicted)
        return self.squared_error
    
    def add_to_visualizer(self,x,y,color='green'):
        plt.scatter(x, y, s=0.5, color=color)
    
    def visualize(self):
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    
    def improvise_state():
        - apply SGD

    def solve(self):
        self.initialize_state_randomly()
        while True:
            self.compute_prediction()
            self.error_fn()
            self.improvise_state()
            self.add_to_visualizer(measurements[0], self.predicted, color='red')
            self.add_to_visualizer(measurements[0], measurements[1], color='blue')
            self.visualize()

if __name__ == '__main__':
    df = DataFactory()
    
    measurements = df.create_linear(1,4,1000,2)

    solver = LeastSquareSolver('Linear', FunctionFactory.linear_function, measurements)

    solver.solve()