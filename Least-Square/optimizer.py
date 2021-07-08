import numpy as np
import matplotlib.pyplot as plt

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
        self.iterations_before_visualization = 150

    def initialize_state_randomly(self):
        self.state_matrix = self.observation_func.initialize_state_matrix()

    def compute_predicted_observation(self):
        self.predicted = self.observation_func.func(self.domain_points, self.state_matrix)

    def compute_error(self):
        self.e_i = self.observation_func.err_fn(self.domain_points, self.observations, self.predicted, self.state_matrix)
        self.squared_error = self.e_i * self.e_i # self.squared_error = np.array(list(map(lambda err: np.dot(np.transpose(err), err), self.e_i)))
        self.sum_of_squared_error = np.sum(self.squared_error)/self.domain_points.shape[0]   
        return self.sum_of_squared_error

    def visualize_loss(self):
        plt.scatter(self.iterations, self.losses, s=0.5, color='green')
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.show()
    
    def improvise_state(self):
        # SGD or Gauss Netwon or Levenberg–Marquardt
        c_i = self.squared_error
        j_i = self.observation_func.jacobian_of_error_func(self.domain_points, self.observations, self.state_matrix)
        b_i = np.matmul(np.moveaxis(self.e_i, 1, 2), j_i)
        h_i = np.array(list(map(lambda j: np.dot(np.transpose(j), j), j_i)))

        c = np.sum(c_i,axis=0)
        b = np.transpose(np.sum(b_i, axis=(0,1)))
        H = np.sum(h_i, axis=0)
        learning_rate = self.observation_func.learning_rate
        change = - learning_rate * np.transpose(np.matmul(np.linalg.inv(H), b))

        self.state_matrix +=  change

    def solve(self):
        self.initialize_state_randomly()
        print (f'The state_matrix is initialed as:\n{np.round(self.state_matrix,decimals=2)}')

        iteration = 0
        self.iterations = []
        self.losses = []
        while True:
            iteration += 1
            self.compute_predicted_observation()
            self.compute_error()
            self.improvise_state()
            self.iterations.append(iteration)
            self.losses.append(self.sum_of_squared_error)
            print (f'Iteration: {iteration}\nL2 error= {np.round(self.sum_of_squared_error,3)}\nstate_matrix = \n {np.round(self.state_matrix,3)} \n\n')
            if iteration%self.iterations_before_visualization == 0:
                self.visualize_loss()
                self.observation_func.run_visualization(self.domain_points, self.observations, self.predicted, self.state_matrix)
