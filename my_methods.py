import numpy as np
from ot import emd



def Psi(func_kernel, data, learnedmat, tau=0):
    n = data.shape[0]
    newlearned = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            newlearned[i,j] = newlearned[j,i] = func_kernel(data[i], data[j], learnedmat) #+ tau * np.linalg.norm(learnedmat, ord=np.inf) * int(i == j) 

    return newlearned / np.linalg.norm(newlearned, ord=np.inf)
class unsupervised_kernel_learning():
    
    def __init__(self, data, func_kernel_A, func_kernel_B, iter_max=1000, regularize=False, regularization_param=None, relative_residual = 1e-3, lipschitz_of_kernel_A=None, lipschitz_of_kernel_B=None, B_0=None):
        self.data = data
        self.func_kernel_A = func_kernel_A
        self.func_kernel_B = func_kernel_B
        self.iter_max = iter_max
        self.regularize = regularize
        self.regularization_param = regularization_param
        self.relative_residual = relative_residual
        self.lipschitz_of_kernel_A = lipschitz_of_kernel_A
        self.lipschitz_of_kernel_B = lipschitz_of_kernel_B
        self.B_0 = B_0

        self.metric, self.B, self.iteration = self.unsupervised_kernel_learning()

    # calculate tau s.t. Psi is definitely contractive.
    def calculate_tau(self, data, lipschitz_const):
        n = len(data)
        norms = np.array([np.linalg.norm(data[i] - data[j], ord=1) for i in range(n) for j in range(i, n)])

        return 2 * lipschitz_const * np.max(np.array(norms)) ** 2

    # Power Iteration Algorithm
    def unsupervised_kernel_learning(self):

        # If Lipschitzconstants are given calculate tau.
        tau_A = self.calculate_tau(self.data, self.lipschitz_of_kernel_A) + self.regularization_param if self.regularize else 0
        tau_B = self.calculate_tau(self.data.T, self.lipschitz_of_kernel_B) + self.regularization_param if self.regularize else 0

        # assume pairwise distance of every point is 1
        B = np.identity(self.data.T.shape[1])
        A = Psi(func_kernel=self.func_kernel_A, 
                data=self.data.T, 
                learnedmat=B, 
                tau=tau_A)


        # power iterations
        for i in range(self.iter_max):
            #print('Iteration: ', i+1)
            old_B = B
            old_A = A

            B = Psi(func_kernel = self.func_kernel_B, 
                    data        = self.data, 
                    learnedmat  = A, 
                    tau         = tau_B)

            A = Psi(func_kernel = self.func_kernel_A, 
                    data        = self.data.T, 
                    learnedmat  = B, 
                    tau         = tau_A)
            
            # check for fixed point
            rel_res_B = np.linalg.norm(old_B - B, ord=np.inf) / np.linalg.norm(B, ord=np.inf)
            rel_res_A = np.linalg.norm(old_A - A, ord=np.inf) / np.linalg.norm(A, ord=np.inf)
            
            if np.max([rel_res_A, rel_res_B]) < self.relative_residual:
            #if np.array_equal(old_B, B) and np.array_equal(old_A, A):
                return A, B, i

        return A, B, self.iter_max

class unsupervised_lin_learning():
    
    def __init__(self, data, iter_max=50, relative_residual=1e-3, B_0=None):
        self.data = data
        self.iter_max = iter_max
        self.relative_residual = relative_residual
        self.B_0 = B_0

        self.metric, self.B, self.iteration = self.unsupervised_lin_learning()

    def Psi_lin_lap(self, data, learnedmat):
        n = data.shape[0]
        newlearned = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dif = data[i] - data[j]
                newlearned[i,j] = newlearned[j,i] = -dif.T @ learnedmat @ dif 
        
        dg = np.sum(newlearned,1)
        newlearned -= np.diag(dg)

        return newlearned/ np.linalg.norm(newlearned, ord=np.inf)

    # Power Iteration Algorithm
    def unsupervised_lin_learning(self):
        B = self.B_0 if self.B_0 else np.identity(self.data.shape[0])
        A = self.Psi_lin_lap(data=self.data.T, learnedmat=B)

        # power iterations
        for i in range(self.iter_max):
            old_B = B
            old_A = A

            B = self.Psi_lin_lap(data=self.data, learnedmat=A)

            A = self.Psi_lin_lap(data=self.data.T, learnedmat=B)
            
            rel_res_B = np.linalg.norm(old_B - B, ord=np.inf) / np.linalg.norm(B, ord=np.inf)
            rel_res_A = np.linalg.norm(old_A - A, ord=np.inf) / np.linalg.norm(A, ord=np.inf)
            
            if np.max([rel_res_A, rel_res_B]) < self.relative_residual:
            #if np.array_equal(old_B, B) and np.array_equal(old_A, A):
                return A, B, i

        return A, B, self.itermax


# project onto cone of (semi-)metric matrices
def triangle_fixing(D, epsilon):
    # ensure that the input is symmetric and has zero diagonal, this might not be the best way to ensure that, see email.
    D = (D + D.T) / 2
    D -= np.diag(np.diag(D))
    
    n = D.shape[0]
    z = np.zeros((n,n,n), dtype=float)
    
    E = np.zeros((n,n), dtype=float)
    
    delta = 1 + epsilon
    
    # progress check
    while delta > epsilon:
        delta = 0

        # iterate over all possible triangle inequalities
        for u in range(n):
            for v in range(u + 1, n):
                for w in range(v + 1, n):
                    triangle = [[u,v,w], [u,w,v], [v,w,u], [v,u,w], [w,u,v], [w,v,u]]
                    
                    # calculate violations and adjust accordingly
                    for i,j,k in triangle:
                        violation = D[i,k] + D[k,j] - D[i,j]
                        theta = (E[i,j] - E[i,k] - E[k,j] - violation) / 3.0
                        theta = max(theta, -z[i,j,k])
                        E[i,j] -= theta
                        E[i,k] += theta
                        E[k,j] += theta
                        z[i,j,k] += theta

                        delta += 3 * np.abs(theta)

    return (D + E) / np.linalg.norm(D + E)
# i used oop because the amount of inputs was getting annoying
class supervised_wasserstein_learning():
    
    def __init__(self, data, labels, sampleSize=50, neighborAmount=3, gradientStep=0.1, p_max=20, q_max=80):
        self.data   = data
        self.labels = labels
        self.n      = len(data)
        self.k      = neighborAmount
        self.step   = gradientStep
        self.p_max  = p_max
        self.q_max  = q_max
        self.EPlus, self.E_iPlus, self.EMinus, self.E_iMinus = self.create_indices()
    

    def create_indices(self):
        EPlus = []
        EMinus = []
        E_iPlus = {i: [] for i in range(self.n)}
        E_iMinus = {i: [] for i in range(self.n)}

        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.labels[i] == self.labels[j]:
                    EPlus.append((i,j))
                    E_iPlus[i].append(j)
                    E_iPlus[j].append(i)

                else:
                    EMinus.append((i,j))
                    E_iMinus[i].append(j)
                    E_iMinus[j].append(i)

        return EPlus, E_iPlus, EMinus, E_iMinus

        
    # calculate k-th nearest neighbor criteria and subgradient. isSim asserts if we look at similar or dissimilar neigbors
    def Algorithm1(self, distanceMat, isSim):
        # we don't precompute the similarity matrix, the omega stays the same but becomes negative if we are looking at dissimilar nbs.
        omega = 1/(distanceMat.shape[0] * self.k) * (1 if isSim else -1)
        E = self.EPlus if isSim else self.EMinus
        E_i = self.E_iPlus if isSim else self.E_iMinus
        optima = {}

        # calculate wasserstein distance w.r.t. to our current distance matrix between all relevant samples 
        for i, j in E:
            if (i, j) not in optima or (j, i) not in optima:
                solutionMat, solutionDic = emd(self.data[i], self.data[j], distanceMat, log=True)
                optima.update({(i, j): [solutionMat, solutionDic['cost']]})
                optima.update({(j, i): [solutionMat, solutionDic['cost']]})

        # calculate subgradient of one the (dis-)similar parts
        G, z = np.zeros((distanceMat.shape)), 0
        for i in range(self.n):
            distancesToNeighbors = [optima[(i,j)] for j in E_i[i]]
            nearestNeighbors = sorted(distancesToNeighbors, key= lambda x: x[1])[:self.k]
            for neighbor in nearestNeighbors:
                G += omega * neighbor[0]
                z += omega * neighbor[1]

        return G, z


    # local linearization of C_k
    def Algorithm2(self):
        # use independence table calculated by algorithm 3
        #D_0 = self.Algorithm3()
        #D_0 = (D_0 + D_0.T) / 2
        #D_0 -= np.diag(np.diag(D_0))
        M_out = np.ones((self.data[0].shape[0], self.data[0].shape[0])) - np.identity(self.data[0].shape[0])

        for p in range(self.p_max):
            gammaPlus, zPlus = self.Algorithm1(distanceMat=M_out, isSim=True)

            M_in = M_out
            for q in range(self.q_max):
                gammaMinus, zMinus = self.Algorithm1(distanceMat=M_in, isSim=False)

                M_in = triangle_fixing(M_in - (self.step / np.sqrt(q + 1)) * (gammaPlus + gammaMinus), 1e-3)

            M_out = M_in
        return M_out


    # use k nearest neighbors and independence tables to find better-than-guessing starting metric
    def Algorithm3(self):
        Xi = np.zeros((self.data[0].shape[0], self.data[0].shape[0]))

        data_mat = [[[] for _ in range(self.n)] for _ in range(self.n)]
        for i in range(self.n):
            for j in range(i + 1, self.n):
                data_mat[i][j] = data_mat[j][i] = [np.linalg.norm(self.data[i] - self.data[j], ord=1), int(self.labels[i] == self.labels[j]), j]
            
        for i, row in enumerate(data_mat):
            N_ikPlus = sorted([x for x in row if x and x[1] == 1], key=lambda x: x[0])[:self.k]
            N_ikMinus = sorted([x for x in row if x and x[1] == 0], key=lambda x: x[0])[:self.k]

            for neighbor in N_ikPlus:
                Xi += 1/(self.n*self.k) * np.outer(self.data[i], self.data[neighbor[-1]])
                
            for neighbor in N_ikMinus:
                Xi -= 1/(self.n*self.k) * np.outer(self.data[i], self.data[neighbor[-1]])

        Xi = triangle_fixing(Xi, 1e-3)

        # regularize s.t. smallest entry is larger than 0
        if min(Xi.flatten()) < 0:
            lam = - 1 / (min(Xi.flatten()) - 1) - 1e-1
            return lam * Xi + (1 - lam) * np.ones(Xi.shape)
        return Xi 
