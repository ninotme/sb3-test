import numpy as np
from gymnasium  import spaces
class HyperPolicy:
    def __init__(self, param_number, alpha=0.5):

        self.alpha = alpha
        self.param_number = param_number

        # Inizializzazione della distribuzione iperparamentrica
        self.mu= [ 0.5 for i in range(self.param_number)]
        self.sigma = [0.5 for i in range(self.param_number)]

        
        print("done initializing")

    # Called by algorithm
    def act(self, state):
        return self.theta
        

    #calcolo del delta_rho
    def eval_gradient(self, actor_params, disc_returns):
        N = len(disc_returns)

        print("actor_params: ", actor_params[0], "len ", len(actor_params[0]))
        sum_sigma = [0 for _ in range(len(actor_params[0])) ]
        
        
        sum_mu = [0 for _ in range( len(actor_params[0]) ) ]


        
        #this can be done in a smarter way TBD once it works
        for i in range(N):
            mus = []
            sigmas = []
            for j in range(len(actor_params[i])):
                theta = actor_params[i][j]
                score_mu = (theta - self.mu[j]) / self.sigma[j] ** 2
                score_sigma = (theta - self.mu[j])**2 - self.sigma[i]/ (self.sigma[j] ** 3)

                #multiplication for the disc
                score_mu = score_mu * disc_returns[i]
                mus.append(score_mu)
                score_sigma = score_sigma * disc_returns[i]
                sigmas.append(score_sigma)
            sum_mu = np.add(sum_mu, mus)
            sum_sigma = np.add(sum_sigma, sigmas)

        grad_mu = np.divide(sum_mu, N)
        grad_sigma = np.divide(sum_sigma, N)
        
        #2 * dim(param_space) vector
        return [grad_mu, grad_sigma]

        pass
        score = []
        for i in range(len(disc_returns)):
            theta_i = actor_params[i]
            diff = np.multiply(np.subtract(theta_i, self.rho), disc_returns[i])
            score.append(diff)

        #calcolo di delta_mu
        delta_rho =[0 for i in range(len(score[0]))]

        for j in range(len(delta_rho)):
            for i in range(len(score)):
                delta_rho[j] = score[i][j] + delta_rho[j]




       # grad = np.sum()
        return np.divide(delta_rho, len(disc_returns))

    def eval_param(self):
        return self.rho
    def set_params(self,rho):
        self.rho = rho
        pass

    def resample(self):
        return [
            np.random.normal(self.mu[i], self.sigma[i]) for i in range(self.param_number)
        ]

    #internal methods
    def set_rho(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
    def get_rho(self):
        return [self.mu, self.sigma]

    def set_theta(self, theta):
        self.theta = theta
    def get_theta(self):
        return self.theta


    ## Unused methods

    def renyi(self, pol):
        pass
    def eval_fisher(self):
        pass

#test con finta reward_batch [1000, 1000]
# il batch di  parametri della policy di test campionati dalla policy
# degli iperparamenteri sono 2:  [theta1, theta2]
if __name__ == '__main__':

    hp = HyperPolicy(3)

    # prelevo due campioni dalla attuale distribuzione degli iperparametri
    theta1 = hp.resample()
    theta2 = hp.resample()

    #test update
    actor_params = [theta1, theta2]
    disc_returns = [1000, 1000]

    # valutazione del gradiente
    grad  = hp.eval_gradient(actor_params, disc_returns)

    #update della distribuzione degli iperparamentri
    step_size = 0.7
    delta_rho = np.multiply(grad, step_size)
    hp.set_params(hp.get_rho() + delta_rho)
    thetaf = hp.resample()
    print('update distribution: ', hp.get_rho[1])
