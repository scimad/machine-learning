import numpy as np

from models import ExpFunction, LinearFunction, SineFunction, QuadraticFunction, ProjectionFunction, SphericalProjection
from optimizer import LeastSquareSolver

if __name__ == '__main__':
    # observation_func = LinearFunction()
    # domain_points, observations = observation_func.get_sample_data(np.ones((1,2)), 1000, noise_sigma=[2])

    # observation_func = QuadraticFunction()
    # domain_points, observations = observation_func.get_sample_data(np.array([[-2,5,2]]), N=1000, noise_sigma=[1.5])

    # observation_func = SineFunction()
    # domain_points, observations = observation_func.get_sample_data(np.array([[1, 2, -3]]), N=1000, noise_sigma=[0.1])

    # observation_func = ExpFunction()
    # domain_points, observations = observation_func.get_sample_data(np.array([[-2, 2]]), N=1000, noise_sigma=[1])
    
    observation_func = SphericalProjection()
    domain_points, observations = observation_func.get_sample_data(20*np.ones((2,3)), N=100, noise_sigma=[0, 0])
    solver = LeastSquareSolver(observation_func, domain_points, observations)
    solver.solve()
