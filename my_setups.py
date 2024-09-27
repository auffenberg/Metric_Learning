import numpy as np
import torch
import wsingular
import time
import my_methods
from torch.utils.tensorboard import SummaryWriter
from metric_learn import LMNN
from metric_learn import ITML


class kernel_method():
    
    def __init__(self, data, RBF_A, RBF_B):
        self.data = data
        self.RBF_A = RBF_A
        self.RBF_B = RBF_B

        self.metric, self.time = self.kernel_method()

    # technically rbf_kernel(sqrt())
    def rbf_kernel_A(self, x, y, mat):
        z = x - y
        dotprod = z.T @ mat @ z
        return np.exp(- dotprod / (2 * self.RBF_A ** 2))

    def rbf_kernel_B(self, x, y, mat):
        z = x - y
        dotprod = z.T @ mat @ z
        return np.exp(- dotprod / (2 * self.RBF_B ** 2))

    def kernel_method(self):
        # iteration parameter to check if fixed point was achieved. if iteration = iter_max most likely no actual fixedpoint
        # func_kernel used for possible choice of other kernels. currently fix rbf_sigma before calling unsupervised_kernel_learning

        start_time = time.time()
        method = my_methods.unsupervised_kernel_learning(data                     = self.data, 
                                                    func_kernel_A            = self.rbf_kernel_A,
                                                    func_kernel_B            = self.rbf_kernel_B,
                                                    regularize               = True,
                                                    lipschitz_of_kernel_A    = 1 / (2 * self.RBF_A ** 2),
                                                    lipschitz_of_kernel_B    = 1 / (2 * self.RBF_B ** 2),
                                                    regularization_param     = 1e-1,
                                                    iter_max                 = 100,
                                                    relative_residual        = 1e-9)
        end_time = time.time()
    
        kernel_time = end_time - start_time
        mahalanobis = method.metric


        return mahalanobis, kernel_time

class linear_method():
    
    def __init__(self, data):
        self.data = data
        self.metric, self.time = self.linear_method()

    def linear_method(self):
        start_time = time.time()
        method = my_methods.unsupervised_lin_learning(data=self.data)
        mahalanobis = method.metric
        lin_time = time.time() - start_time

        return mahalanobis, lin_time

class unsupervised_wasserstein_method():
    def __init__(self, data):
        self.data = data
        self.metric, self.time = self.unsupervised_wasserstein_method()

    def unsupervised_wasserstein_method(self):
        writer = SummaryWriter()

        start_time = time.time()
        # Wasserstein singular vectors from Peyre's paper
        C, D = wsingular.sinkhorn_singular_vectors(dataset      = torch.tensor(self.data), 
                                                eps          = 5e-2, 
                                                dtype        = torch.double, 
                                                device       = 'cpu',
                                                n_iter       = 100,
                                                tau          = 1e-1,
                                                writer       = writer,
                                                progress_bar = True) 

        end_time = time.time()
        wasserstein_time = end_time - start_time


        return np.array(C), wasserstein_time

class euclidean_method():
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.metric = np.identity(len(data[0]))
        self.time = 0

    
class supervised_wasserstein_method():
    
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.metric, self.time = self.supervised_wasserstein_method()

    def supervised_wasserstein_method(self):
        start_time = time.time()
        method = my_methods.supervised_wasserstein_learning(data            = self.data,
                                        labels          = self.labels,
                                        neighborAmount  = 3,
                                        gradientStep    = 0.1,
                                        p_max           = 20,
                                        q_max           = 20)
        ground_metric = method.Algorithm2()

        end_time = time.time() - start_time

        return ground_metric, end_time

    
class LMNN_method():
    
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.metric, self.time = self.LMNN_method()

    def LMNN_method(self):
        start_time = time.time()
        lmnn = LMNN(n_neighbors=5, learn_rate=1e-6)
        mahalanobis = lmnn.fit(self.data, self.labels).get_mahalanobis_matrix()
        end_time = time.time() - start_time

        return mahalanobis, end_time

    
class ITML_method():
    
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.metric, self.time = self.ITML_method()

    def ITML_method(self):
        (n, _) = self.data.shape
        
        data_for_method = []
        similarity_for_method = []
        for i in range(n):
            for j in range(i+1, n):
                data_for_method.append([self.data[i], self.data[j]])
                if self.labels[i] == self.labels[j]:
                    similarity_for_method.append(1)
                else:
                    similarity_for_method.append(-1)
        
        start_time = time.time()
        itml = ITML()
        itml = itml.fit(data_for_method, similarity_for_method)
        end_time = time.time() - start_time
        components = itml.components_
        mahalanobis = components.T @ components

        return mahalanobis, end_time