import numpy as np
import matplotlib.pyplot as plt
from numpy.__config__ import show


class FunctionTemplate:
    def __init__(self, learning_rate=1):
        self.learning_rate = learning_rate

    def func(self, *args):
        '''
        Should return predicted observations
        '''
        pass

    def visualize(self, x, y,color=['red'],show=True):
        if len(color) == 1:
            plt.scatter(x,y,marker='x',s=1, color=color)
        else:
            for x_,y_,c in zip(x,y,color):
                plt.scatter(x_, y_ , marker='o', s=10, color=c)
        plt.xlabel('x (data point)')
        plt.ylabel('y (observations)')
        if show:
            plt.show()

    def err_fn(self, domain_points, observations, predicted, state_matrix):
        e_i = np.reshape(observations, np.shape(predicted)) - predicted
        return e_i

    def run_visualization(self, domain_points, observations, predicted, state_matrix):
        self.visualize(domain_points, observations, color='g', show=False)
        self.visualize(domain_points, predicted, color='r')


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
        super().__init__(learning_rate=0.005)

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



class SphericalProjection(FunctionTemplate):
    '''
    2D simulation:

    The actual 3d position of features on images are shown by X, Y

    (x,y) represents the position of that feature on 360 image.

    The features in two sequences are ordered by correspondence.
    i.e. x[0],y[0] and X[0]Y[0] both have same (or almost same) features

    The center of (x, y) and theta will vary and that for X, Y will remain constant

    The variation of (x,y) will be by varying the angle theta.


    x,y represents domain points and corresponding to every point, there will be a line for each point passing through the center


    '''
    def __init__(self):
        super().__init__(learning_rate=0.1)
        self.sample_data = None

    def func(self, domain_points, state_matrix):
        pass

    def initialize_state_matrix(self):
        # return np.average(self.sample_data[0]), np.average(self.sample_data[1]), 2*np.pi*np.random.rand()
        return np.average(self.sample_data[0]), np.average(self.sample_data[1]), 2

    def get_sample_data(self, state_matrix=np.ones((2,3)), N=1000, noise_sigma=[1,2]):
        self.N = N
        #let feature_vec represent the feature discriptors of keypoints that match
        # on both the 360 image and on the image dataset
        feature_vec = np.random.uniform(0, 1, (N,1,3))

        # let alpha represent the angles observed by different feature points in 360 image
        # alpha = np.random.uniform(0, 2 * np.pi, (N, 1, 1))
        alpha = np.random.exponential(1, size=(N, 1, 1))
        alpha = alpha %( 2 * np.pi)

        print (f'Computed sample alpha:\n {alpha}')

        self.x = 1*np.cos(alpha) # + np.random.normal(size=alpha.shape)
        self.y = 1*np.sin(alpha) # + np.random.normal(size=alpha.shape)

        # These are the parameters we would later want our solver to find
        CX, CY, THETA = state_matrix

        # Location of features in 2d space (images (represented by lines) in 2d)
        noise_in_data = np.random.uniform(0, 0.03*2*np.pi, np.shape(alpha)) #meaning % of 2*pi
        angles_in_real_world = alpha +  THETA + noise_in_data
        X = CX + np.random.uniform(3, 4, (N, 1, 1)) * np.cos(angles_in_real_world)
        Y = CY + np.random.uniform(3, 4, (N, 1, 1)) * np.sin(angles_in_real_world)
        F_V = np.copy(feature_vec)

        self.feature_vec = feature_vec

        self.visualize(self.x, self.y, feature_vec, show=False)
        self.visualize(X, Y, F_V)

        self.sample_data = np.stack((X, Y))
        self.sample_observations = alpha
        return self.sample_data, self.sample_observations

    def err_fn(self, domain_points, observations, predicted, state_matrix):
        x_i, y_i = domain_points[0], domain_points[1]
        alpha_i = observations
        cx, cy, theta = state_matrix
        yicy = y_i - cy
        xicx = x_i - cx
        d2_i = yicy**2 + xicx**2
        beta_i = []
        for y, x in zip(yicy, xicx):
            beta_i.append(np.arctan2(y,x))
        beta_i = np.array(beta_i)
        gamma_i = (beta_i - alpha_i - theta)
        e_i = d2_i*np.sin(gamma_i) #*np.sin(gamma_i)
        return e_i

    def jacobian_of_error_func(self, domain_points, observations, state_matrix):
        x_i, y_i = domain_points[0], domain_points[1]
        alpha_i = observations
        cx, cy, theta = state_matrix
        yicy = y_i - cy
        xicx = x_i - cx

        beta_i = []
        for y, x in zip(yicy, xicx):
            beta_i.append(np.arctan2(y,x))
        beta_i = np.array(beta_i)

        gamma_i = beta_i - alpha_i - theta
        d2 = (xicx**2 + yicy**2)

        st = np.sin(gamma_i)
        ct = np.cos(gamma_i)

        x_st = xicx*st
        y_ct = yicy*ct
        x_ct = xicx*ct
        y_st = yicy*st

        # When the error function is d2_i*np.sin(gamma_i)*np.sin(gamma_i)
        j_11 = -2*x_st + y_ct
        j_12 = -2*y_st - x_ct
        j_13 = - d2 * ct

        # When the error function is d2_i*np.sin(gamma_i)*np.sin(gamma_i)
        # j_11 = 2*st*(-x_st + y_ct)
        # j_12 = -2*st*(x_ct + y_st)
        # j_13 = - 2* d2 * st * ct

        j_i = np.concatenate((j_11, j_12, j_13), axis=2)
        return j_i

    def  run_visualization(self, domain_points, observations, predicted, state_matrix):
        X, Y  = self.sample_data[0], self.sample_data[1]
        F_V = self.feature_vec
        cx, cy, theta = state_matrix
        alpha = self.sample_observations
        x = cx + 1*np.cos(alpha + theta)
        y = cy + 1*np.sin(alpha + theta)
        self.visualize(x, y, self.feature_vec, show=False)
        self.visualize(X, Y, F_V)
