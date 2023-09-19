import numpy as np
from gymnasium  import spaces

forced_policy = False
verbose = True

class HyperPolicy:
    def __init__(self, param_number, alpha=0.5):

        self.alpha = alpha
        self.param_number = param_number

        # Inizializzazione della distribuzione iperparamentrica
        self.mu= [ 1 for i in range(self.param_number)]
        self.sigma = [0.7 for i in range(self.param_number)]

        self.theta = []
        
        print("done initializing")

    # Called by algorithm
    #this should be overwritten by a policy 
    def act(self, state):
        print("act::state[0]: ", state[0])
        if forced_policy: 
            return 0
        
        feature1 =  np.multiply(self.theta[0], state[0]) 
        feature2 =  np.multiply(self.theta[1], state[0])
        
        print("feature1 = ", self.theta[0], " x ", state[0])
        print("feature2 = ", self.theta[1], " x ", state[0]) 
        print("feature1 = ", feature1)
        print("feature2 = ", feature2) 
        
        if feature1 > 3 and feature2 < 6 : 
            
            # go left (in the direction of the goal) 
            return 0
        else:
            return 1
        

    #calcolo del delta_rho
    def eval_gradient(self, actor_params, disc_returns, use_baseline=False):
        N = len(disc_returns)

        print("actor_params: ", actor_params[0], "len ", len(actor_params[0]))
        sum_sigma = [0 for _ in range(len(actor_params[0])) ]
        
        
        sum_mu = [0 for _ in range( len(actor_params[0]) ) ]


        print("-------in eval_gradient()________") 
        print("disc_returns = ", disc_returns) 
        print("actor_params = ", actor_params) 
        

        
        #this can be done in a smarter way TBD once it works
        for i in range(N):
            mus = []
            sigmas = []
            for j in range(len(actor_params[i])):
                theta = actor_params[i][j]
                
                score_mu = (theta - self.mu[j]) / self.sigma[j] ** 2
                
                
                score_sigma = ((theta - self.mu[j])**2 - (self.sigma[j]**2)) / (self.sigma[j] ** 3)
                
                
                #multiplication for the disc
                score_mu = score_mu * disc_returns[i]
                mus.append(score_mu)
                
                
                score_sigma = score_sigma * disc_returns[i]
                sigmas.append(score_sigma)
                
                if verbose: 
                    print("theta(", j, ") = ", theta)
                    print("score_mu = ", score_mu)
                    print("score_sigma = ", score_sigma)
                    print("mus = ", mus) 
                    print("sigmas = ", sigmas) 
                    
            sum_mu = np.add(sum_mu, mus)
            
            
            sum_sigma = np.add(sum_sigma, sigmas)
            
            if verbose: 
                print("sum_mu = ", sum_mu) 
                print("sum_sigma = ", sum_sigma) 
                
        grad_mu = np.divide(sum_mu, N)
        grad_sigma = np.divide(sum_sigma, N)
        
        if verbose: 
            print('[', grad_mu, ', \n', grad_sigma, ']' ) 
        
        #2 * dim(param_space) vector
        
       
        
        print("grad_sigma = ", grad_sigma) 
        print("self.sigma = ", self.sigma) 
        return [grad_mu, grad_sigma]

       

    def eval_params(self):
        return [ self.mu, self.sigma] 
    
    def set_params(self,rho):
        self.mu = rho[0] 
        self.sigma = rho[1]
        

    def resample(self):
        self.theta = [
            np.random.normal(self.mu[i], (self.sigma[i] ** 2)) for i in range(self.param_number)
        ]
        return self.theta

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

    hp = HyperPolicy(2)

    # prelevo due campioni dalla attuale distribuzione degli iperparametri
    theta1 = hp.resample()
    theta2 = hp.resample()

    #test update
    actor_params = [theta1, theta2]
    disc_returns = [1000, 1000]

    # valutazione del gradiente
    grad  = hp.eval_gradient(actor_params, disc_returns)

    #update della distribuzione degli iperparamentri
    step_size = 0.0001
    delta_rho = np.multiply(grad, step_size)
    hp.set_params(hp.get_rho() + delta_rho)
    thetaf = hp.resample()
    print('update distribution: ', hp.get_rho()[0], hp.get_rho()[1]) 
